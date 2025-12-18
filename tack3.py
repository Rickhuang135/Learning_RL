import torch
import torch.nn as nn
import torch.optim as optim

from tack_board import *
from tack_nn import A2CModel
from tack_ultils import find_model
from tack_ultils import is_end
from device import *
from datetime import datetime
import time
import json
import numpy as np

from tack_benchmark import benchmark

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)
torch.serialization.add_safe_globals([A2CModel])

MODELPATH = "./tack_models/"

class BoardDataset(torch.utils.data.Dataset):
    def __init__(self, rb: ReplayBuffer):
        self.states = torch.tensor(np.concat([generate_symmetries(s) for s in rb.states]), dtype = torch.float32).to(device)
        self.actions = torch.tensor(np.concat([generate_symmetries(a) for a in rb.actions]), dtype = torch.float32).to(device)
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, ind):
        return self.states[ind], self.actions[ind]

class EpisodeData:
    def __init__(self):
        self.Pi_loss = torch.zeros(1, device=device)
        self.V_loss = torch.zeros(1, device=device)
        self.entropy_loss = torch.zeros(1, device=device)

    def __repr__(self):
        return f"Pi_loss: {self.Pi_loss.item():.5f} \t V_loss: {self.V_loss.item():.5f} \t Entropy_loss {self.entropy_loss.item():.5g} "

    def __str__(self):
        return self.__repr__()

                
class Train:
    def __init__(self, hyper_params: dict):
        self.model = A2CModel().to(device)

        self.r_win = 1
        self.r_draw = 0

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
    
    def backprop_with_symmetries(self, model: A2CModel, rb: ReplayBuffer , winner: int|None = None, episode_loss: EpisodeData | None = None):
        dataset=BoardDataset(rb)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        model.train()
        for states, actions in dataloader:
            self.optimiser.zero_grad()
            logits, values = model(states)
            rotation_period = len(values)//len(rb)
            values_means = torch.sum(values.reshape(-1, rotation_period), dim=-1)
            if winner is not None:
                V_last_state = winner * self.r_win
                value_labels = torch.clone(values_means).detach()
                value_labels[-1] = V_last_state
                Vloss = self.criterionV(values_means, value_labels)
            else:
                value_labels = values_means[1:].detach() * self.gamma
                Vloss = self.criterionV(values_means[:-1], value_labels)

            As = (values_means[1:]-values_means[:-1]).detach()
            As = As.repeat(rotation_period, 1).transpose(1,0).flatten()
            log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
            valid_actions = actions[:-rotation_period] # filtering out AMNone value
            valid_log_prob = log_prob[:-rotation_period]
            Piloss = (-1* valid_log_prob[valid_actions!=0] * As).sum()
            
            prob = torch.nn.functional.softmax(logits, dim=-1)
            entropy = -(prob*log_prob).sum(dim=1).mean()
            entropy_loss = -1 * self.entropy_beta * entropy
            
            (Vloss+Piloss+entropy_loss).backward()
            self.optimiser.step()
            
            if episode_loss is not None:
                episode_loss.Pi_loss+=torch.abs(Piloss)
                episode_loss.V_loss+=torch.abs(Vloss)
                episode_loss.entropy_loss+=entropy_loss
        
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
    
def train_loop(
        episodes = 1000,
):  
    hyper_params = {
        'gamma':0.95,
        'entropy_beta':0.03,
        'learn_rate': 0.0001,
        'replay_length': 3,
        'parallel_games': 10,
        # 'model_prefix': '12_12_1245_tack3',
    }
    train = Train(hyper_params)
    print_period = min(episodes//10, 50)
    replay_length = hyper_params["replay_length"]
    parallel_games = hyper_params["parallel_games"]
    no_r = 3

    game_uid = parallel_games
    states = new_states = np.zeros((parallel_games,9)) # array of flat board states
    game_ids = np.arange(parallel_games, dtype=np.int32)
    is_first_player_turn = np.ones(parallel_games, dtype = np.bool) # flat array of 1s
    rewards = None
    rb = ReplayBuffer(states, game_ids, is_first_player_turn)
    begin_time = time.time()
    while game_uid-parallel_games < episodes:
        for i in range(replay_length):
            move_inds = train.infer(states) # get moves from inference
            new_states = np.copy(states) # get new states from moves
            new_states[np.arange(parallel_games),move_inds] = (is_first_player_turn-0.5)*2
            for i, state in enumerate(new_states.reshape(-1,3,3)): # for each new state:
                end = is_end(state)
                if end!=-1:
                    if rewards is None:
                        rewards = np.zeros(parallel_games, dtype=np.int8)+no_r
                    rewards[i] = (is_first_player_turn[i]-0.5)*2 if end==1 else 0 # player who just moved wins
                    game_uid += 1
                    game_ids[i] = game_uid
                    new_states[i] = np.zeros(9) # replace with empty board
            is_first_player_turn= np.invert(is_first_player_turn)
            if rewards is not None:
                is_first_player_turn[rewards!=no_r] = True
            rb.append(new_states=new_states, game_ids=game_ids, is_first_player_turn= is_first_player_turn, new_actions=move_inds, rewards= rewards)
            rewards = None
            states = new_states
        print(rb.to_df())
        
        raise Exception("stop for a moment")
        train.backprop_with_symmetries(train.model, rb)
        rb.empty()


        # for j in range(hyper_params["replay_length"]):
            # get moves from inference
            # get new states from moves
            # for each new state:
            #   if is end state:
            #       add reward to reward buffer
            #       increment episodes_completed
            #       replace with new empty board
            # Add new states and moves into replay buffer
            # flip is_first_player_turn

        # back propegate with replay buffer
        # empty replay buffer    


        # loss = train.episode()
        # if i % print_period == 0:
        #     print(f"{round(i/episodes*100)}% {loss}")
    time_elapsed = time.time()-begin_time
    steps_per_second = train.steps/time_elapsed
    print(f"{train.steps} steps of {hyper_params["replay_length"]*7} completed in {time_elapsed:.3f} seconds at {steps_per_second:.3f} steps/second")

    # benchmark(lambda x, id :train.a1(x)*id)

    if episodes >= 2000: # save the model
        print(f"Model saved to {train.save(increment_version=True, extra_info={
            "episodes": episodes,
            "time_elapsed": time_elapsed, 
            "steps/second": steps_per_second,
            })}")

    return train

train_res=train_loop()

# test_positions = torch.tensor([
#     [1,-1,0,
#     0,0,0,
#     0,0,0],

#     [1,-1,0,
#     0,1,0,
#     0,0,0],

#     [1,-1,0,
#     0,1,0,
#     0,0,-1],

#     [1,-1,0,
#     0,1,0,
#     1,0,-1],

#     [1,-1,0,
#     -1,1,0,
#     1,0,-1],
# ], device=device, dtype=torch.float32)

# logits, values = train_res.model(test_positions)
# probs = torch.nn.functional.softmax(logits, -1)
# from tack_ultils import pt
# for prob, value, position in zip(probs, values, test_positions):
#     pt(position)
#     pt(prob)
#     print(value)
# play(lambda board, id: train_res.a1(board) * id)

# components:
# 1. Board
# 2. Reward giving environment
# 3. Agent which uses Va values to make moves and update V
# 4. Model(s) which gives Va values

# value system:
# while not s0 end:
    # make move at s0, creating s1
    # if s1 is end state
    #   update V1 with draw or win
    # 
    # update V0 with s0
    # update Pi with V0
    # s0 = s1