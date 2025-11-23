import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tack_board import *
from tack_nn import DeepQModel
from tack_nn import PolicyModel
from tack_ultils import pt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision = 3)

class EpisodeData:
    def __init__(self):
        self.Pi_loss =0
        self.Q_loss = 0
        self.total_rp = 0
        self.total_ro = 0
    
    def flip(self):
        self.total_rp, self.total_ro = self.total_ro, self.total_rp

    def r_both(self, value):
        self.total_rp+=value
        self.total_ro+=value

    def __repr__(self):
        return f"Pi_loss: {self.Pi_loss:.3f} \t Q_loss: {self.Q_loss:.3f} \t total_rewards: {self.total_rp:>3}, {self.total_ro:>5}"

    def __str__(self):
        return self.__repr__()

class Agent:
    def __init__(self, id: int, Q: DeepQModel, Pi:PolicyModel):
        self.Q = Q
        self.Pi = Pi
        self.id = id # id can not be zero

    def infer(self, board: Board):
        self.Pi.eval()
        with torch.no_grad():
            raw_dist: torch.Tensor = self.Pi(board.state)
            raw_dist[(board.legal_moves.flatten()==0)]=0
            legal_dist = raw_dist/torch.sum(raw_dist)
            cum_dist = legal_dist.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1).to(device)).to(device)
            AM: torch.Tensor = torch.zeros(9).to(device)
            AM[idx]=1
            return AM.reshape((3,3)), legal_dist
                
class Train:
    def __init__(self):
        self.Q = DeepQModel().to(device)
        self.Pi = PolicyModel().to(device)
        self.a1 = Agent(1, self.Q, self.Pi)
        # self.a2 = Agent(1, self.Q, self.Pi)

        self.r_win = 0
        self.r_lose = 20
        self.r_draw = 1
        self.r_move = 1

        self.gamma = 0.8
        self.Psi_discount = 0.01
        self.criterionQ = nn.MSELoss()
        self.optimiserQ = optim.Adam(self.Q.parameters(),lr=0.0005)
        self.criterionPi = nn.CrossEntropyLoss()
        self.optimiserPi = optim.Adam(self.Pi.parameters(),lr=0.001)

        self.verbose = False

            # print(f"\n\nInstance of Q_backprop\n")
            # print(f"s: \n{s}")
            # print("Qa")
            # pt(Qa)
            # print(f"label:")
            # pt(label)
            # print(f"loss: {loss}")

        # if self.verbose:
        #     print("\n\nInstance of Pi_backprop\n")
        #     print(f"s: \n{s}")  
        #     print("Pis")
        #     pt(Pis)
        #     print("AM")
        #     pt(AM)
        #     print("raw_label")
        #     pt(Pi0)
        #     print(f"Psi: \n{Psi.cpu().item()}")
        #     print("label")
        #     pt(label)
        #     print(f"loss: {loss}")
    
    def backprop_with_symmetries(self, Q: DeepQModel, Pi: PolicyModel, s0: Board, AM0: torch.Tensor, real_reward, epsiode_loss: EpisodeData | None = None, s2: Board | None = None):
        Q.train()
        Pi.train()
        s0_s = s0.generate_symmetries()
        AM0_s = generate_symmetries(AM0)
        AM0_s = [ AM.flatten() for AM in AM0_s]
        if s2 is not None:
            s2_s = s2.generate_symmetries()
        total_Pi_loss =0
        total_Q_loss =0
        for index, (s0, AM0) in enumerate(zip(s0_s, AM0_s)):
            # update Q
            if s2 is not None:
                Q.eval()
                Pi.eval()
                with torch.no_grad():
                    Pis2 = Pi(s2_s[index].state) # type:ignore
                    Qs2 = Q(s2[index].state) # type:ignore
                Q.train()
                Pi.train()
                expected_Qs2 = torch.sum(Pis2* Qs2)
                value = real_reward + self.gamma*expected_Qs2
            else:
                value = real_reward
            self.optimiserQ.zero_grad()
            Qs0 = Q(s0.state)
            Qlabel = torch.clone(Qs0)
            Qlabel[AM0==1] = value
            Qloss = self.criterionQ(Qs0, Qlabel)
            Qloss.backward()
            self.optimiserPi.step()
            total_Q_loss += Qloss
            
            # update Pi with Psi
            Qs0 = Qs0.detach()
            self.optimiserPi.zero_grad()
            Pis0 = Pi(s0.state)
            V0 = torch.sum(Qs0 * Pis0)
            Psi = Qs0[AM0 == 1] - V0
            Pilabel = torch.clone(Pis0)
            Pilabel[AM0==1] += Psi*self.Psi_discount
            Piloss =self.criterionPi(Pis0, Pilabel)
            Piloss.backward()
            self.optimiserPi.step()
            total_Pi_loss+=Piloss.item()

        if epsiode_loss is not None:
            epsiode_loss.Pi_loss+=total_Pi_loss
            epsiode_loss.Q_loss+=total_Q_loss

    def episode(self):
        loss = EpisodeData()

        s0 = Board()
        player = self.a1
        opp = player
        # opp = self.a2
        AM0, Pi0 = player.infer(s0)
        s1= s0.next(AM0)
        while not s1.end:
            AM1, Pi1 = opp.infer(s1)
            s2 = s1.next(AM1)
            player.Q.eval()
            if s2.end: # board ends
                if s2.winner == 0: # draw
                    loss.r_both(self.r_draw)
                    self.backprop_with_symmetries(player.Q, player.Pi, s0, AM0, self.r_draw, loss)
                    self.backprop_with_symmetries(opp.Q,opp.Pi, s1,  AM1, self.r_draw, loss)
                else:
                    loss.total_rp += self.r_lose
                    loss.total_ro += self.r_win
                    self.backprop_with_symmetries(player.Q, player.Pi, s0, AM0, self.r_lose, loss)
                    self.backprop_with_symmetries(opp.Q,opp.Pi, s1, AM1,self.r_win, loss)
            else:
                self.backprop_with_symmetries(self.Q,self.Pi, s0, AM0, self.r_move, loss)
                loss.total_rp+= self.r_move
                del AM0, Pi0
                AM0, Pi0 = AM1, Pi1
                # player, opp = opp, player
                loss.flip()
            
            del AM1, s0, Pi1
            s0 = s1
            s1 = s2
        del s1, AM0
        return loss
    
def train_loop(
        episodes = 500,
):
    
    train = Train()

    for i in range(episodes):
        loss = train.episode()
        if i % 7 == 0:
            print(f"{round(i/episodes*100)}% {loss}")
        if episodes-i <= 2:
            train.verbose=True
    return train

a1=train_loop().a1
play(lambda board: a1.infer(board)[0])


# begin
# make actual move at s0 using Pi0, creating s1

# while not end:
# opponent makes theoretical move at s1 using Pi1, creating s2
#     if board ends with draw, 
#       update Q with [draw] at s0
#     else:
#       [loss] at s0
#       [win] at s1
# inference to get Qa2 at s2
# update Q at s0 using Qa2

# s1 becomes s0, s2 becomes s1
# Qa1 becomes Qa0, Qa2 becomes Qa1


# components:
# 1. Board
# 2. Reward giving environment
# 3. Agent which uses Qa values to make moves and update Q
# 4. Model(s) which gives Qa values