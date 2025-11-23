import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tack_board import *
from tack_nn import DeepQModel
from tack_nn import PolicyModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision = 3)

def pt(matflat: torch.Tensor):
    mat3x3=matflat.reshape(3,3)
    print(mat3x3.detach().cpu().numpy())

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

    def backprop(self, Q: DeepQModel, Pi: PolicyModel, s: Board, AM: torch.Tensor, Pi0: torch.Tensor, value, Qa0):
        Q.train()
        states= s.generate_symmetries()
        AMs=generate_symmetries(AM)
        lossPi= lossQ =0
        for state, position in zip(states,AMs):
            position = position.flatten()
            lossQ+= self.Q_backprop(Q,state,position,value)
            lossPi+=self.Pi_backprop(Pi, s, position, Pi0, Qa0-value)
        return  lossQ, lossPi

    def Q_backprop(self, Q: DeepQModel, s: Board, AM: torch.Tensor, value):
        self.optimiserQ.zero_grad()
        Qa = Q(s.state)
        label = torch.clone(Qa)
        label[AM!=0]=value
        loss= self.criterionQ(Qa, label)
        loss.backward()
        self.optimiserQ.step()
        if self.verbose:
            print(f"\n\nInstance of Q_backprop\n")
            print(f"s: \n{s}")
            print("Qa")
            pt(Qa)
            print(f"label:")
            pt(label)
            print(f"loss: {loss}")
        return loss.item()

    def Pi_backprop(self, Pi: PolicyModel, s: Board, AM: torch.Tensor, Pi0:torch.Tensor, Psi):
        self.optimiserPi.zero_grad()
        Pis = Pi(s.state)
        label = torch.clone(Pi0)
        label[AM!=0]+=Psi*self.Psi_discount
        loss = self.criterionPi(Pis, Pi0)
        loss.backward()
        self.optimiserPi.step()
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
        return loss.item()

    def episode(self):
        Q_loss = 0.0
        Pi_loss = 0.0
        tr_p = 0 # total reward for player
        tr_o = 0 # total reward for other

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
            Qa0 = player.Q(s0.state)[AM0.flatten()==1]
            if s2.end: # board ends
                if s2.winner == 0: # draw
                    tr_p += self.r_draw
                    tr_o += self.r_draw
                    tmp=self.backprop(player.Q, player.Pi, s0, AM0, Pi0, self.r_draw,Qa0)
                    Q_loss += tmp[0]
                    Pi_loss += tmp[1]
                    tmp=self.backprop(opp.Q,opp.Pi, s1,  AM1, Pi1,self.r_draw, Qa0)
                    Q_loss += tmp[0]
                    Pi_loss += tmp[1]
                else:
                    tr_p += self.r_lose
                    tr_o += self.r_win
                    tmp=self.backprop(player.Q, player.Pi, s0, AM0, Pi0,self.r_lose, Qa0)
                    Q_loss += tmp[0]
                    Pi_loss += tmp[1]
                    tmp=self.backprop(opp.Q,opp.Pi, s1, AM1, Pi1,self.r_win,Qa0)
                    Q_loss += tmp[0]
                    Pi_loss += tmp[1]
            else:
                s2s = s2.generate_symmetries()
                s0s = s0.generate_symmetries()
                Pi0s = generate_symmetries(Pi0.reshape(3,3))
                Pi2s = [player.infer(s)[1] for s in s2s]
                Qa2s = [player.Q(s.state).detach() for s in s2s]
                AM0s = generate_symmetries(AM0)
                tr_p += self.r_move
                player.Q.train()
                player.Pi.train()
                for Qa2, AM, s, Pi2, Pi0 in zip(Qa2s, AM0s, s0s, Pi2s, Pi0s):
                    AM=AM.flatten()
                    Pi0=Pi0.flatten()
                    expected_Q2=torch.sum(Qa2*Pi2)
                    Psi = Qa0 - expected_Q2
                    Pi_loss += self.Pi_backprop(player.Pi, s, AM, Pi0, Psi)
                    Q_loss+=self.Q_backprop(player.Q, s, AM, self.r_move+self.gamma*expected_Q2) # using expected SARSA
                del AM0, Pi0
                AM0, Pi0 = AM1, Pi1
                # player, opp = opp, player
                tr_p,tr_o = tr_o, tr_p
            
            del AM1, s0, Pi1
            s0 = s1
            s1 = s2
        del s1, AM0
        return Pi_loss, Q_loss, (tr_p, tr_o)
    
    
def train_loop(
        episodes = 10,
):
    
    train = Train()

    for i in range(episodes):
        Pi_loss, Q_loss, (tr_1, tr_2) = train.episode()
        if i % 7 == 0:
            print(f"{round(i/episodes*100)}% Pi_loss: {Pi_loss:.3f} \t Q_loss: {Q_loss:.3f} \t total_rewards: {tr_1:>3}, {tr_2:>5}")
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