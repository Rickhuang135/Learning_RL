import torch
import torch.nn as nn
import torch.optim as optim

from tack_board import play
from tack_board import Board
from tack_nn import DeepQModel
from tack_nn import PolicyModel
import numpy as np
from random import choice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(sci_mode=False)
    
class MCTS:
    c = np.sqrt(2) # exploration parameter

    def __init__(self, action, head):
        self.head:MCTS =head
        self.action = action
        self.visits=1
        self.children: list[MCTS]=[]
        # implement code to get reward
        self.reward: int|None = None

    def UCB(self):
        if self.reward is None:
            raise Exception("Node not finished, cannot evaluate")
        return self.reward + self.c*np.sqrt(np.log(self.head.visits)/self.visits)

    def next(self, actions: torch.Tensor):
        new_acts = []
        existing_acts=[child.action for child in self.children]
        for act in actions:
            if not act in existing_acts:
                new_acts.append(act)
        if len(new_acts):
            return MCTS(choice(new_acts), self)
        UCBs= np.array([child.UCB() for child in self.children])
        return self.children[np.argmin(UCBs)]

    # run from starting state to end
        # at starting state, gets list of actions
        # check for new action, take if exist
        # check which child has lowest UCB
        # repeat until new action is reached 
    # when creating new leaf
        # randomly make moves until terminal state reach
        # create new child with reward and visit count
    # create leaf based on first action and reward

class Agent:
    def __init__(self, id: int, Q: DeepQModel, Pi:PolicyModel):
        self.Q = Q
        self.Pi = Pi
        self.id = id # id can not be zero

    def infer(self, board: Board):
        self.Pi.eval()
        raw_dist: torch.Tensor = self.Pi(board.state)
        raw_dist[(board.legal_moves.flatten()==0)]=0
        legal_dist = raw_dist/torch.sum(raw_dist)
        cum_dist = legal_dist.cumsum(0)
        idx = torch.searchsorted(cum_dist, torch.rand(1))
        AM: torch.Tensor = torch.zeros(9)
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
        self.criterionQ = nn.MSELoss()
        self.optimiserQ = optim.Adam(self.Q.parameters(),lr=0.0005)
        self.criterionPi = nn.CrossEntropyLoss()
        self.optimiserPi = optim.Adam(self.Pi.parameters(),lr=0.001)

        self.verbose = False

    def backprop(self, Q: DeepQModel, s: Board, AM: torch.Tensor, value):
        Q.train()
        states= generate_symmetries(s.state)
        AMs=generate_symmetries(AM)
        loss =0
        for state, position in zip(states,AMs):
            loss+= self.backprop_ind(Q,state,position,value)
        return loss

    def backprop_ind(self, Q: DeepQModel, s: torch.Tensor, AM: torch.Tensor, value):
        self.optimiserQ.zero_grad()
        Qa = Q(s)
        label = torch.clone(Qa)
        label[AM!=0]=value
        loss= self.criterionQ(Qa, label)
        loss.backward()
        self.optimiserQ.step()
        if self.verbose:
            print(f"s: \n{s}")
            print(f"Qa: \n{Qa}")
            print(f"label: \n{label}")
            print(f"loss: {loss}")
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
            if s2.end: # board ends
                if s2.winner == 0: # draw
                    tr_p += self.r_draw
                    tr_o += self.r_draw
                    Q_loss+=self.backprop(player.Q, s0, AM0, self.r_draw)
                    Q_loss+=self.backprop(opp.Q,s1, AM1, self.r_draw)
                else:
                    tr_p += self.r_lose
                    tr_o += self.r_win
                    Q_loss+=self.backprop(player.Q, s0, AM0, self.r_lose)
                    Q_loss+=self.backprop(opp.Q,s1, AM1, self.r_win)
            else:
                s2s = generate_symmetries(s2.state)
                s0s = generate_symmetries(s0.state)
                Pi2s = [player.infer(s) for s in s2s]
                Qa2s = [player.Q(s).detach() for s in s2s]
                AM0s = generate_symmetries(AM0)
                tr_p += self.r_move
                for Qa, AM,s in zip(Qa2s, AM0s,s0s):
                    AM=AM.flatten()
                    Q_loss+=self.backprop_ind(player.Q,s, AM, self.r_move+self.gamma*torch.sum(Qa)) # using expected SARSA
                del AM0
                AM0 = AM1
                # player, opp = opp, player
                tr_p,tr_o = tr_o, tr_p
            Psi = player.Q(s0)[AM0==1]-torch.sum(player.Q(s1)*player.Pi(s1)) # how much the state value changed based on the Policy move
            del AM1
            del s0
            s0 = s1
            s1 = s2
        del s1
        del AM0
        return Pi_loss, Q_loss, (tr_p, tr_o)
    
def generate_symmetries(mat3x3: torch.Tensor) -> list[torch.Tensor]:
    results = []
    results.append(torch.clone(mat3x3))
    results.append(torch.flip(mat3x3,[1,0]))
    results.append(torch.fliplr(mat3x3))
    results.append(torch.flipud(mat3x3))
    results.append(torch.transpose(mat3x3,1,0))
    return [r.flatten() for r in results]

def train_loop(
        episodes = 400,
):
    
    train = Train()

    for i in range(episodes):
        Pi_loss, Q_loss, (tr_1, tr_2) = train.episode()
        if i % 7 == 0:
            print(f"Pi_loss: {Pi_loss:.3g} \t Q_loss: {Q_loss:.3g} \t total_rewards: {tr_1:>3}, {tr_2:>5}")
        if episodes-i <= 3:
            train.verbose=True
    return train

# def train_loop2():
#     s0 = Board()
#     player = Agent(1, DeepQModel())
#     head = MCTS(0, None)
#     actions = torch.arange(9)[s0.legal_moves.flatten()==1]
#     current_node = head.next(actions)
#     while not current_node.reward is None:
#         current_node=current_node.next(s0)
#     total_reward = 0
#     while not s0.end:
#         s0.write(player.infer(s0,2))
#         total_reward+=1
#         if s0.end:
#             current_node.reward=20
        


# train_loop2()
a1=train_loop().a1
play(a1.infer)


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