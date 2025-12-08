import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tack_board import *
from tack_nn import ValueModel
from tack_nn import PolicyModel
from tack_ultils import pt
from device import device

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)
np.set_printoptions(precision = 3)

class EpisodeData:
    def __init__(self):
        self.Pi_loss =0
        self.V_loss = 0
        self.total_rp = 0
        self.total_ro = 0
    
    def flip(self):
        self.total_rp, self.total_ro = self.total_ro, self.total_rp

    def r_both(self, value):
        self.total_rp+=value
        self.total_ro+=value

    def __repr__(self):
        return f"Pi_loss: {self.Pi_loss:.3f} \t V_loss: {self.V_loss:.3f} \t total_rewards: {self.total_rp:>3}, {self.total_ro:>5}"

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
            # pt(raw_output)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim=0)
            cum_dist = prob.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1).to(device)).to(device)
            AM: torch.Tensor = torch.zeros(9).to(device)
            AM[idx]=1
            return AM.reshape((3,3))*self.id
                
class Train:
    def __init__(self):
        self.V = ValueModel().to(device)
        self.Pi = PolicyModel().to(device)
        self.a1 = Agent(1, self.Pi)
        self.a2 = Agent(-1, self.Pi)
        # self.a2 = lambda x: infer(x)*-1

        self.r_win = 1
        self.r_draw = 0

        self.gamma = 0.9
        self.Psi_discount = 0.002
        self.criterionV = nn.MSELoss()
        self.optimiserV = optim.Adam(self.V.parameters(),lr=0.0005)
        self.optimiserPi = optim.Adam(self.Pi.parameters(),lr=0.001)

        self.verbose = False
    
    def backprop_with_symmetries(self, V: ValueModel, Pi: PolicyModel, s0: Board, s1: Board, AM0: torch.Tensor, epsiode_loss: EpisodeData | None = None):
        V.train()
        Pi.train()
        s0_s = s0.generate_symmetries()
        AM0_s = generate_symmetries(AM0)
        AM0_s = [ AM.flatten() for AM in AM0_s]
        s1_s = s1.generate_symmetries()
        total_Pi_loss =0
        total_V_loss =0
        for index, (s0, s1, AM0) in enumerate(zip(s0_s, s1_s, AM0_s)):
            # update V
            if s1.end:
                Vs1 = s1.winner * (self.r_win - s1.depth*0.01) * torch.ones(1, dtype=torch.double).to(device) #type:ignore
            else:
                Vs1 = V(s1.state).detach()
            self.optimiserV.zero_grad()
            Vs0 = V(s0.state)
            Vlabel = Vs1*self.gamma
            Vloss = self.criterionV(Vs0, Vlabel)
            Vloss.backward()
            self.optimiserV.step()
            total_V_loss += Vloss.item()
            
            # update Pi with Advantage function
            A = (Vs1 * self.gamma - Vs0.detach())*AM0[AM0!=0]
            self.optimiserPi.zero_grad()
            logits = Pi(s0)
            log_prob = torch.nn.functional.log_softmax(logits, dim=0)
            Piloss = -1*log_prob[AM0!=0]*A

            prob = torch.nn.functional.softmax(logits, dim=0)
            entropy = -prob*log_prob.mean()
            loss_entropy = -1 * entropy * self.Psi_discount

            Piloss.backward(retain_graph = True)
            # Piloss.backward()
            loss_entropy.mean().backward()

            self.optimiserPi.step()
            torch.nn.utils.clip_grad_norm_(Pi.parameters(),0.1)
            total_Pi_loss+=Piloss.item()
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

                # Pi.eval()
                # logits = Pi(s0)
                # log_prob = torch.nn.functional.log_softmax(logits, dim=0)
                # prob = torch.nn.functional.softmax(logits, dim=0)
                # entropy = -prob*log_prob.mean()
                # print("Policy")
                # pt(prob)
                # print("Log prob")
                # pt(log_prob)
                # print("Entropy")
                # pt(entropy)
                # Pi.train()

            break
        if epsiode_loss is not None:
            epsiode_loss.Pi_loss+=total_Pi_loss
            epsiode_loss.V_loss+=total_V_loss

    def episode(self):
        loss = EpisodeData()

        s0 = Board()
        s0.write(torch.Tensor([
    [1,-1,0],
    [1,0,0],
    [0,0,0],
    ]))
        player = self.a2
        opp = self.a1
        while not s0.end:
            AM0 = player(s0)
            s1= s0.next(AM0)
            self.backprop_with_symmetries(self.V,self.Pi, s0, s1,AM0, loss)
            player, opp = opp, player
            del AM0, s0
            s0 = s1
        return loss
    
def train_loop(
        episodes = 1000,
):  
    train = Train()

    for i in range(episodes):
        if episodes-i <= 1:
            train.verbose=True
        loss = train.episode()
        if i % 5 == 0:
            print(f"{round(i/episodes*100)}% {loss}")
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