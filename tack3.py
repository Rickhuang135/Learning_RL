import torch
import torch.nn as nn
import torch.optim as optim

from tack_board import *
from tack_nn import DeepQModel
import numpy as np
from random import randint
from random import random
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
    def __init__(self, id: int, Q: DeepQModel):
        self.Q = Q
        self.id = id # id can not be zero

    def infer(self, board: Board, epsilon: float = 0.0):
        self.Q.eval()
        if random() < epsilon:
            Qa = torch.randperm(9).to(device)
        else:
            with torch.no_grad():
                Qa = self.Q(board.state)
        Qa+= (board.legal_moves.flatten()==0)*100
        min_a = torch.min(Qa)
        AM =(Qa==min_a) # AM is short for "addition matrix"
        n_moves = len(AM[AM])
        if n_moves != 1:
            ord=torch.randperm(n_moves).to(device)
            Qa[AM]+= ord
            AM = (Qa==min_a)
        return AM.reshape(3,3)*self.id
                
class Train:
    def __init__(self):
        self.Q = DeepQModel().to(device)
        self.a1 = Agent(1, self.Q)
        self.a2 = Agent(1, self.Q)

        self.r_win = 0
        self.r_lose = 20
        self.r_draw = 1
        self.r_move = 1

        self.epsilon = 1.1
        self.gamma = 0.8
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(self.Q.parameters(),lr=0.0005)

        self.verbose = False

    def backprop(self, Q: DeepQModel, s: Board, AM: torch.Tensor, value):
        Q.train()
        states= s.generate_symmetries()
        AMs=generate_symmetries(AM)
        loss =0
        for state, moves in zip(states,AMs):
            loss+= self.backprop_ind(Q,state,moves.flatten(),value)
        return loss

    def backprop_ind(self, Q: DeepQModel, s: Board, AM: torch.Tensor, value):
        self.optimiser.zero_grad()
        Qa = Q(s.state)
        label = torch.clone(Qa)
        label[AM!=0]=value
        loss= self.criterion(Qa, label)
        loss.backward()
        self.optimiser.step()
        if self.verbose:
            print(f"s: \n{s}")
            print(f"Qa: \n{Qa.reshape(3,3)}")
            print(f"label: \n{label.reshape(3,3)}")
            print(f"loss: {loss}")
        return loss.item()

    def episode(self, epsilon: float):
        running_loss = 0.0
        tr_p = 0 # total reward for player
        tr_o = 0 # total reward for other

        s0 = Board()
        player = self.a1
        opp = self.a2
        AM0 = player.infer(s0, epsilon)
        s1= s0.next(AM0)
        while not s1.end:
            AM1 = opp.infer(s1, epsilon)
            s2 = s1.next(AM1)
            if s2.end: # board ends
                if self.verbose:
                    print(f"s0: \n{s0}")
                    print(f"s1: \n{s1}")
                    print(f"s2: \n{s2}")
                if s2.winner == 0: # draw
                    tr_p += self.r_draw
                    tr_o += self.r_draw
                    running_loss+=self.backprop(player.Q,s0, AM0, self.r_draw)
                    running_loss+=self.backprop(opp.Q,s1, AM1, self.r_draw)
                else:
                    tr_p += self.r_lose
                    tr_o += self.r_win
                    running_loss+=self.backprop(player.Q, s0, AM0, self.r_lose)
                    running_loss+=self.backprop(opp.Q,s1, AM1, self.r_win)
            else:
                s2s = s2.generate_symmetries()
                s0s = s0.generate_symmetries()
                Qas = [player.Q(s.state).detach() for s in s2s]
                AM2s = [player.infer(s,epsilon=0) for s in s2s]
                target_values = [self.r_move+self.gamma*Qai[AM2i.flatten()!=0] for Qai, AM2i in zip(Qas,AM2s)]
                AM0s = generate_symmetries(AM0)
                tr_p += self.r_move
                if self.verbose:
                    print(f"s0: \n{s0}")
                    print(f"s1: \n{s1}")
                    print(f"s2: \n{s2}")
                    print(f"Qa0: \n{player.Q(s0.state).reshape(3,3)}")
                    print(f"Qa0: \n{player.Q(s1.state).reshape(3,3)}")
                    print(f"Qa2: \n{torch.clone(Qas[0]).reshape(3,3)}")
                    print(f"target value: {target_values[0]}")
                for AM,s, target_value in zip(AM0s, s0s, target_values):
                    AM = AM.flatten()
                    running_loss+=self.backprop_ind(player.Q,s, AM, target_value)
                del AM0
                AM0 = AM1
                player, opp = opp, player
                tr_p,tr_o = tr_o, tr_p
            del AM1
            del s0
            s0 = s1
            s1 = s2
        del s1
        del AM0
        return running_loss, (tr_p, tr_o)

def train_loop(
        episodes = 10,
        epsilon = 1.0
):
    interval = episodes//10
    step_size = epsilon/10
    if interval==0:
        interval = 1
    
    train = Train()

    for i in range(episodes):
        running_loss, (tr_1, tr_2) = train.episode(epsilon)
        if i % 7 == 0:
            print(f"epsilon: {epsilon:1f} \t runningloss during episode: {running_loss:.5g} \t total_rewards: {tr_1:>3}, {tr_2:>5}")
        if episodes-i <= 3:
            train.verbose=True
        if i % interval ==0:
            epsilon -= step_size
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
# make actual move at s0 using Qa0, creating s1

# while not end:
# opponent makes theoretical best move/random move at s1 using Qa1, creating s2
    # if board ends with draw, update Q with 
    #   [draw] at s0
    # else:
    #   [loss] at s0
    #   [win] at s1
# inference to get Qa2 at s2
# update Q at s0 using Qa2

# s1 becomes s0, s2 becomes s1
# Qa1 becomes Qa0, Qa2 becomes Qa1


# components:
# 1. Board
# 2. Reward giving environment
# 3. Agent which uses Qa target_values to make moves and update Q
# 4. Model(s) which gives Qa target_values