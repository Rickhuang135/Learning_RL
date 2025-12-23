from device import device
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
from replayBuffer import ReplayBuffer
from ultils import *
from datetime import datetime
from nn import A2CModel
from board import Board
from lossCollector import LossCollector
torch.serialization.add_safe_globals([A2CModel])
                
class Train:
    def __init__(self, hyper_params: dict):
        self.model = A2CModel().to(device)
        self.hyper_params = hyper_params
        if "model_prefix" in hyper_params.keys():
            model_paths, version = find_model(MODELPATH,hyper_params["model_prefix"])
            self.version = version
            for model_path in model_paths:
                if "model" in model_path:
                    print(f"loading model {model_path} of version {self.version}")
                    self.model=torch.load(MODELPATH+model_path,map_location=device, weights_only=False)
        self.gamma = hyper_params['gamma']
        self.entropy_beta = hyper_params['entropy_beta']
        self.criterionV = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(),lr=hyper_params['learn_rate'])
        self.replay_buffer_length = hyper_params['replay_length']
        self.steps = 0
        self.verbose = False

    def infer(self, states_flat: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(torch.tensor(states_flat, dtype=torch.float32).to(device), Pi_only = True)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim = -1)
            cum_dist = prob.cumsum(-1)
            idx = torch.searchsorted(cum_dist, torch.rand(states_flat.shape[0], device=device).reshape(-1,1))
        return idx.detach().cpu().numpy().flatten()
    
    def backprop_with_symmetries(self, model: A2CModel, rb: ReplayBuffer, lc: LossCollector | None = None):
        # retrieve data from replay buffer
        training_shape = (rb.n_parallel, self.replay_buffer_length-1, symmetry_generator.n_ops)
        states, game_ids, is_first_player_turns, actions, rewards = rb.get_all() 

        # Send data to GPU
        states_torch = torch.as_tensor(states.reshape(-1,9), dtype =torch.float32, device=device) # (n_parallel, replay_length, 9) -> (n_parallel × replay_length, 9)
        actions_tensors = torch.as_tensor(actions[:,:-1], dtype=torch.int32, device=device) # (n_parallel, replay_length - 1)
        player_ids = torch.as_tensor((is_first_player_turns[:,:-1]-0.5)*2, device=device) # (n_parallel, replay_length - 1)
        rewards_torch = torch.as_tensor(rewards[:,:-1], dtype=torch.int8, device=device) # (n_parallel, replay_length - 1)
        
        # augmenting data
        augmented_states = symmetry_generator.rotate(states_torch) # (n_parallel × replay_length, 9) -> (n_parallel x replay_length, n_symmetries, 9)
        augmented_actions = symmetry_generator.rotate_indicies(actions_tensors).permute(0,2,1) # (n_parallel, replay_length-1) -> (n_parallel, n_symmetries, replay_length-1) -> (n_parallel, replay_length-1, n_symmetries)
        augmented_rewards = torch.broadcast_to(rewards_torch.unsqueeze(-1), training_shape) #(n_parallel, replay_length - 1, n_symmetries)
        augmented_player_ids = torch.broadcast_to(player_ids.unsqueeze(-1), training_shape) # (n_parallel, replay_length -1 ) -> (n_parallel, replay_length-1, n_symmetries)
        augmented_game_ids = torch.broadcast_to(torch.as_tensor(game_ids[:,:-1]).unsqueeze(-1), training_shape) # (n_parallel, replay_length) -> (n_parallel, replay_length-1, n_symmetries)

        # infering with model
        model.train()
        self.optimiser.zero_grad()
        logits, values = model(torch.Tensor(augmented_states.reshape(-1,9)).to(device)) # logits are flat
        values: torch.Tensor = values.reshape(rb.n_parallel, -1, symmetry_generator.n_ops) # (n_parallel x replay_length × n_symmetries) -> (n_parallel, replay_length, n_symmetries)
        logits_sanatised = logits.reshape(rb.n_parallel, self.replay_buffer_length, symmetry_generator.n_ops, 9)[:,:-1] # (n_parallel x replay_length × n_symmetries, 9) -> (n_parallel, replay_length-1, n_symmetries, 9)
        log_prob = torch.nn.functional.log_softmax(logits_sanatised, dim=-1) # (n_parallel, replay_length-1, n_symmetries, 9)
        prob = torch.nn.functional.softmax(logits_sanatised, dim=-1) # (n_parallel, replay_length-1, n_symmetries, 9)

        # calculate V loss
        Vlabels = values[:, 1:].detach() # (n_parallel, replay_length - 1, n_symmetries)
        Vlabels: torch.Tensor = torch.where(augmented_rewards==r_none, Vlabels, augmented_rewards)*self.gamma
        trainable_values = values[:,:-1] # (n_parallel, replay_length - 1, n_symmetries)
        Vloss= self.criterionV(trainable_values, Vlabels)

        # calculate Policy loss
        advantage = Vlabels - trainable_values.detach() # (n_parallel, replay_length - 1, n_symmetries)
        actions_expanded = augmented_actions[:,:].unsqueeze(-1) # (n_parallel, replay_length, n_symmetries) -> (n_parallel, replay_length-1 , n_symmetries, 1)  add nested layer to match log_prob dimensions 
        valid_log_prob = log_prob.gather(dim=3, index=actions_expanded).squeeze(-1) # (n_parallel, replay_length-1, n_symmetries)
        grad_Pi = valid_log_prob*advantage*augmented_player_ids # (n_parallel, replay_length-1, n_symmetries)
        Piloss = torch.sum(grad_Pi)

        # entropy normalisation
        entropy = -(prob*log_prob).sum()
        entropy_loss = -1 * self.entropy_beta * entropy

        # append data to logs
        if lc is not None:
            # detach and reshape data
            trainable_states = augmented_states.reshape(rb.n_parallel, self.replay_buffer_length, symmetry_generator.n_ops, 9)[:, :-1]
            lc.append(
                inputs=[trainable_states, augmented_actions, augmented_player_ids, augmented_rewards, augmented_game_ids], 
                outputs=[trainable_values.detach()]+[Vlabels, advantage]+[x.detach() for x in [ prob, valid_log_prob, grad_Pi,]], 
                loss=[x.detach() for x in [Piloss, Vloss, entropy_loss]]
                )
        
        # back-propagate and step
        (Vloss+Piloss+entropy_loss).backward()
        self.optimiser.step()
        self.steps += 1

    
    def save(self, increment_version = True, extra_info: dict = {}):
        if 'model_prefix' in self.hyper_params.keys():
            if increment_version:
                self.version[1]+=1
            prefix = self.hyper_params['model_prefix']
            version_str = '_'.join(str(x) for x in (self.version))
            saved_paths = [
                f"{MODELPATH}/{prefix}_model#{version_str}.pt",
                f"{MODELPATH}/{prefix}#{version_str}.json",
            ]
        else:
            now = datetime.now().strftime('%m_%d_%H%M')
            saved_paths = [
                f"{MODELPATH}{now}_tack3_model#0_0.pt",
                f"{MODELPATH}{now}_tack3#0_0.json",
            ]
        torch.save(self.model, saved_paths[0])
        info_dict = {
            'device': str(device)
        }
        info_dict.update(self.hyper_params)
        info_dict.update({ # type:ignore
            'steps': self.steps,  # type:ignore
        })
        info_dict.update(extra_info)
        with open(saved_paths[1], "w") as file:
            json.dump(info_dict, file, indent=4)
        return saved_paths

    def benchmark_handler(self, board: Board, id:int):
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(torch.tensor((board.state.flatten()), dtype=torch.float32).to(device), Pi_only = True)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim=0)
            cum_dist = prob.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1, device=device))
            AM: torch.Tensor = torch.zeros_like(prob)
            AM[idx]=1
            return AM.detach().cpu().numpy().reshape((3,3))*id
    