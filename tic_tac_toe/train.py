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
        self.gamma = hyper_params['gamma']
        self.entropy_beta = hyper_params['entropy_beta']
        self.criterionV = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(),lr=hyper_params['learn_rate'])
        self.replay_buffer_length = hyper_params['replay_length']
        if "model_prefix" in hyper_params.keys():
            model_paths, version = find_model(MODELPATH,hyper_params["model_prefix"])
            self.version = version
            for model_path in model_paths:
                if "model" in model_path:
                    print(f"loading model {model_path} of version {self.version}")
                    self.model=torch.load(MODELPATH+model_path,map_location=device, weights_only=False)
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
        # retrieve data from replay bufferlc.append()
        states, game_ids, is_first_player_turns, actions, rewards = rb.get_all() 
        augmented_states = symmetry_generator.rotate(states.reshape((-1,9))) # augment with symmetries added in new dimension

        # Sending data to GPU
        player_ids = torch.tensor((is_first_player_turns-0.5)*2).to(device)
        augmented_actions = symmetry_generator.rotate_indicies(actions).transpose((1,0,2))
        action_tensors = torch.tensor(augmented_actions).to(device) # The start of each row is the original action
        rewards_torch = torch.tensor(rewards[:,:-1], dtype=torch.int8).to(device)

        # infering with model
        model.train()
        self.optimiser.zero_grad()
        logits, values = model(torch.Tensor(augmented_states.reshape(-1,9)).to(device)) # logits are flat
        values: torch.Tensor = values.reshape(rb.n_parallel, -1, symmetry_generator.n_ops)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).reshape(rb.n_parallel, -1, symmetry_generator.n_ops, 9)
        prob = torch.nn.functional.softmax(logits, dim=-1).reshape(rb.n_parallel, -1, symmetry_generator.n_ops, 9)

        # calculate V loss
        mean_values = torch.mean(values, dim=2)
        Vlabels = mean_values[:, 1:].detach()
        Vlabels = torch.where(rewards_torch==r_none, Vlabels, rewards_torch)*self.gamma
        trainable_values = mean_values[:,:-1]
        Vloss= self.criterionV(trainable_values, Vlabels)

        # calculate Policy loss
        advantage = Vlabels - trainable_values.detach()
        actions_expanded = action_tensors[:,:-1].unsqueeze(-1) # add nested layer to match log_prob dimensions
        valid_log_prob = log_prob[:,:-1].gather(dim=3, index=actions_expanded).squeeze(-1)
        grad_Pi = valid_log_prob.mean(dim=2)*advantage*player_ids[:,:-1]
        Piloss = torch.sum(grad_Pi)
        # print(valid_log_prob)
        # print(valid_log_prob.shape)

        # entropy normalisation
        entropy = -(prob*log_prob).sum()
        entropy_loss = -1 * self.entropy_beta * entropy
        (Vloss+Piloss+entropy_loss).backward()
        self.optimiser.step()

        if lc is not None:
            lc.append(game_ids, prob, grad_Pi, values, Vlabels, Piloss, Vloss, entropy_loss)
        self.steps += 1

    
    def save(self, increment_version = True, extra_info: dict = {}):
        if 'model_prefix' in self.hyper_params.keys():
            if increment_version:
                self.version[1]+=1
            prefix = self.hyper_params['model_prefix']
            version_str = '_'.join(str(x) for x in (self.version))
            saved_paths = [
                f"./{MODELPATH}/{prefix}_model#{version_str}.pt",
                f"./{MODELPATH}/{prefix}#{version_str}.json",
            ]
        else:
            now = datetime.now().strftime('%m_%d_%H%M')
            saved_paths = [
                f"./{MODELPATH}/{now}_tack3_model#0_0.pt",
                f"./{MODELPATH}/{now}_tack3#0_0.json",
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
    