import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tack_board import *
from tack_nn import ValueModel
from tack_nn import PolicyModel
from tack_ultils import pt
from tack_ultils import find_model
from device import *
from datetime import datetime
import time
import json

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)
np.set_printoptions(precision = 3)

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
    def __init__(self, id: int, Pi:PolicyModel):
        self.Pi = Pi
        self.id = id # id can not be zero

    def __call__(self, board):
        self.Pi.eval()
        with torch.no_grad():
            raw_output = self.Pi(board)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim=0)
            cum_dist = prob.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1, device=device))
            AM: torch.Tensor = torch.zeros_like(prob)
            AM[idx]=1
            return AM.reshape((3,3))*self.id
                
class Train:
    def __init__(self, hyper_params: dict):
        self.V = ValueModel().to(device)
        self.Pi = PolicyModel().to(device)
        self.a1 = Agent(1, self.Pi)
        self.a2 = Agent(-1, self.Pi)

        self.r_win = 1
        self.r_draw = 0

        self.hyper_params = hyper_params
        self.gamma = hyper_params['gamma']
        self.entropy_beta = hyper_params['entropy_beta']
        self.criterionV = nn.MSELoss()
        self.optimiserV = optim.Adam(self.V.parameters(),lr=hyper_params['V_learn_rate'])
        self.optimiserPi = optim.Adam(self.Pi.parameters(),lr=hyper_params['Pi_learn_rate'])
        if "model_prefix" in hyper_params.keys():
            model_paths, version = find_model(MODELPATH,hyper_params["model_prefix"])
            self.version = version
            for model_path in model_paths:
                if "V_model" in model_path:
                    self.V.load_state_dict(torch.load(MODELPATH+model_path, weights_only=True))
                elif "Pi_model" in model_path:
                    self.Pi.load_state_dict(torch.load(MODELPATH+model_path, weights_only=True))
        self.steps = 0
        self.verbose = False
    
    def backprop_with_symmetries(self, V: ValueModel, Pi: PolicyModel, s0: Board, s1: Board, AM0: torch.Tensor, episode_loss: EpisodeData | None = None):
        V.train()
        Pi.train()
        s0_s = s0.generate_symmetries()
        AM0_s = generate_symmetries(AM0)
        AM0_s = [ AM.flatten() for AM in AM0_s]
        s1_s = s1.generate_symmetries()
        total_Pi_loss = torch.zeros(1, device=device)
        total_V_loss = torch.zeros(1, device=device)
        total_entropy_loss = torch.zeros(1, device=device)
        for index, (s0, s1, AM0) in enumerate(zip(s0_s, s1_s, AM0_s)):
            # update V
            if s1.end:
                Vs1 = s1.winner * (self.r_win - s1.depth*0.01) * torch.ones(1, dtype=torch.float32, device=device) #type:ignore
            else:
                Vs1 = V(s1.state).detach()
            self.optimiserV.zero_grad()
            Vs0 = V(s0.state)
            Vlabel = Vs1*self.gamma
            Vloss = self.criterionV(Vs0, Vlabel)
            Vloss.backward()
            self.optimiserV.step()
            total_V_loss += Vloss
            
            # update Pi with Advantage function
            A = (Vs1 * self.gamma - Vs0.detach())*AM0[AM0!=0]
            self.optimiserPi.zero_grad()
            logits = Pi(s0)
            log_prob = torch.nn.functional.log_softmax(logits, dim=0)
            Piloss = -1*log_prob[AM0!=0]*A

            prob = torch.nn.functional.softmax(logits, dim=0)
            entropy = -prob*log_prob.mean()
            loss_entropy = (-1 * entropy * self.entropy_beta).mean()

            Piloss.backward(retain_graph = True)
            # Piloss.backward()
            loss_entropy.backward()

            self.optimiserPi.step()
            torch.nn.utils.clip_grad_norm_(Pi.parameters(),0.1)
            total_entropy_loss+= loss_entropy
            total_Pi_loss+=Piloss
            if index==0 and self.verbose:
                # V information
                print(f"\n\nInstance of V_backprop\n")
                print(f"board: \n{s0}")
                print("Vs0: Value at state 0")
                pt(Vs0)
                print("move")
                pt(AM0)
                print("Vs1: Value at state 1")
                pt(Vs1)
                print("Vlabel")
                pt(Vlabel)
                print(f"loss: {Vloss.cpu().item()}")

                # Pi information
                print("\nInstance of Pi_backprop\n")
                print(f"board: \n{s0}")
                print("Policy")
                pt(prob)
                print("Log prob")
                pt(log_prob)
                print("move")
                pt(AM0)
                print("advantage")
                pt(A)
                print("Piloss:")
                pt(Piloss)
                # print("Entropy")
                # pt(entropy)
                # print("Entropy loss")
                # pt(loss_entropy)

            break
        if episode_loss is not None:
            episode_loss.Pi_loss+=abs(total_Pi_loss)
            episode_loss.V_loss+=total_V_loss
            episode_loss.entropy_loss+=total_entropy_loss
        
        self.steps += 1

    def episode(self):
        loss = EpisodeData()

        s0 = Board()
        s0.write(torch.Tensor([
    [1,-1,0],
    [0,0,0],
    [0,0,0],
    ]).to(device))
        player = self.a1
        opp = self.a2
        while not s0.end:
            AM0 = player(s0)
            s1= s0.next(AM0)
            self.backprop_with_symmetries(self.V,self.Pi, s0, s1,AM0, loss)
            player, opp = opp, player
            del AM0, s0
            s0 = s1
        return loss
    
    def save(self, increment_version = True, extra_info: dict = {}):
        if 'model_prefix' in self.hyper_params.keys():
            if increment_version:
                self.version[1]+=1
            prefix = self.hyper_params['model_prefix']
            version_str = '_'.join(str(x) for x in (self.version))
            saved_paths = [
                f"./{MODELPATH}/{prefix}_V_model#{version_str}.pt",
                f"./{MODELPATH}/{prefix}_Pi_model#{version_str}.pt",
                f"./{MODELPATH}/{prefix}#{version_str}.json",
            ]
        else:
            now = datetime.now().strftime('%m_%d_%H%M')
            saved_paths = [
                f"./{MODELPATH}/{now}_tack3_V_model#0_0.pt",
                f"./{MODELPATH}/{now}_tack3_Pi_model#0_0.pt",
                f"./{MODELPATH}/{now}#0_0.json",
            ]
        torch.save(self.V.state_dict(), saved_paths[0])
        torch.save(self.Pi.state_dict(), saved_paths[1])
        info_dict = {
            'device': str(device)
        }
        info_dict.update(self.hyper_params)
        info_dict.update({ # type:ignore
            'steps': self.steps,  # type:ignore
        })
        info_dict.update(extra_info)
        with open(saved_paths[2], "w") as file:
            json.dump(info_dict, file, indent=4)
        return saved_paths
    
def train_loop(
        episodes = 500,
):  
    hyper_params = {
        'gamma':0.95,
        'entropy_beta':0.0022,
        'V_learn_rate': 0.0005,
        'Pi_learn_rate': 0.0007,
        'model_prefix': '12_10_1622_tack3',
    }
    train = Train(hyper_params)
    print_period = min(episodes//10, 50)

    begin_time = time.time()
    for i in range(episodes):
        # if episodes-i <= 1:
        #     train.verbose=True
        loss = train.episode()
        if i % print_period == 0:
            print(f"{round(i/episodes*100)}% {loss}")
    time_elapsed = time.time()-begin_time
    steps_per_second = train.steps/time_elapsed
    print(f"{train.steps} steps completed in {time_elapsed:.3f} seconds at {steps_per_second:.3f} steps/second")

    if episodes >= 50: # save the model
        print(f"Model saved to {train.save(increment_version=False, extra_info={
            "episodes": episodes,
            "time_elapsed": time_elapsed, 
            "steps/second": steps_per_second,
            })}")

    return train

a1=train_loop().a1
# play(lambda board: a1.infer(board)[0])


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