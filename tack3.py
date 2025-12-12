import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tack_board import *
from tack_nn import A2CModel
from tack_ultils import find_model
from device import *
from datetime import datetime
import time
import json

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)
np.set_printoptions(precision = 3)
torch.serialization.add_safe_globals([A2CModel])

MODELPATH = "./tack_models/"

class EpisodeData:
    def __init__(self):
        self.Pi_loss = torch.zeros(1, device=device)
        self.V_loss = torch.zeros(1, device=device)
        self.entropy_loss = torch.zeros(1, device=device)

    def __repr__(self):
        return f"Pi_loss: {self.Pi_loss.item():.5f} \t V_loss: {self.V_loss.item():.5f} \t Entropy_loss {self.entropy_loss.item():.5g} "

    def __str__(self):
        return self.__repr__()

class Agent:
    def __init__(self, id: int, model: A2CModel):
        self.model = model
        self.id = id # id can not be zero

    def __call__(self, board: Board):
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(extract_state(board), Pi_only = True)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim=0)
            cum_dist = prob.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1, device=device))
            AM: torch.Tensor = torch.zeros_like(prob)
            AM[idx]=1
            return AM.reshape((3,3))*self.id
                
class Train:
    def __init__(self, hyper_params: dict):
        self.model = A2CModel().to(device)
        self.a1 = Agent(1, self.model)
        self.a2 = Agent(-1, self.model)

        self.r_win = 1
        self.r_draw = 0

        self.hyper_params = hyper_params
        self.gamma = hyper_params['gamma']
        self.entropy_beta = hyper_params['entropy_beta']
        self.criterionV = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(),lr=hyper_params['learn_rate'])
        self.replay_buffer_length = hyper_params['replay_buffer_length']
        if "model_prefix" in hyper_params.keys():
            model_paths, version = find_model(MODELPATH,hyper_params["model_prefix"])
            self.version = version
            for model_path in model_paths:
                if "model" in model_path:
                    print(f"loading model {model_path} of version {self.version}")
                    self.model=torch.load(MODELPATH+model_path,map_location=device, weights_only=False)
        self.steps = 0
        self.verbose = False
    
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
                V_last_state = winner * (self.r_win  - rb.depth*0.01)
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
        
        self.steps += len(dataset) -1 
        # self.steps += 1

    def episode(self):
        loss = EpisodeData()
        s0 = Board()
        rb = ReplayBuffer(s0)
        s0.write(torch.Tensor([
    [1,-1,0,
    0,0,0,
    0,0,0],
    ]).reshape(3,3).to(device))
        player = self.a1
        opp = self.a2
        while not s0.end:
            if len(rb) == self.replay_buffer_length:
                self.backprop_with_symmetries(self.model, rb, episode_loss=loss)
                rb.empty()
            AM0 = player(s0)
            s1= s0.next(AM0)
            rb.append(s1,AM0)
            player, opp = opp, player
            del AM0, s0
            s0 = s1

        if len(rb) > 1:
            self.backprop_with_symmetries(self.model, rb, winner=s0.winner, episode_loss=loss)
        return loss
    
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
        'replay_buffer_length': 5,
        # 'model_prefix': '12_12_1245_tack3',
    }
    train = Train(hyper_params)
    print_period = min(episodes//10, 50)

    begin_time = time.time()
    for i in range(episodes):
        if episodes-i <= 2:
            train.verbose=True
        loss = train.episode()
        if i % print_period == 0:
            print(f"{round(i/episodes*100)}% {loss}")
    time_elapsed = time.time()-begin_time
    steps_per_second = train.steps/time_elapsed
    print(f"{train.steps} steps completed in {time_elapsed:.3f} seconds at {steps_per_second:.3f} steps/second")

    if episodes >= 1000: # save the model
        print(f"Model saved to {train.save(increment_version=True, extra_info={
            "episodes": episodes,
            "time_elapsed": time_elapsed, 
            "steps/second": steps_per_second,
            })}")

    return train

train_res=train_loop()

test_positions = torch.tensor([
    [1,-1,0,
    0,0,0,
    0,0,0],

    [1,-1,0,
    0,1,0,
    0,0,0],

    [1,-1,0,
    0,1,0,
    0,0,-1],

    [1,-1,0,
    0,1,0,
    1,0,-1],

    [1,-1,0,
    -1,1,0,
    1,0,-1],
], device=device, dtype=torch.float32)

logits, values = train_res.model(test_positions)
probs = torch.nn.functional.softmax(logits, -1)
from tack_ultils import pt
for prob, value, position in zip(probs, values, test_positions):
    pt(position)
    pt(prob)
    print(value)
play(lambda board, id: train_res.a1(board) * id)

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