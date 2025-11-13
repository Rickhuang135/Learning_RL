import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from random import randint
from random import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(sci_mode=False)
class Board:
    def __init__(self, isclone=False):
        if not isclone:
            self.state =torch.tensor(np.zeros((3,3))).to(device)
            self.legal_moves=torch.where(self.state==0,1,0)
            self.end =False
            self.winner = 0
            
    def __str__(self):
        return f"{self.state}"

    def write(self, addition_matrix):
        self.state+=addition_matrix
        self.legal_moves[addition_matrix!=0]=0
        value = addition_matrix[addition_matrix!=0][0] # assumes one move at a time
        match self.is_end():
            case 1:
                self.end = True
                self.winner = value
            case 0:
                self.end = True
                self.winner = 0
            case -1:
                self.end = False
        
    def next(self, addition_matrix):
        result = self.copy()
        result.write(addition_matrix)
        result.state*=-1
        return result

    def copy(self):
        b=Board(True)
        b.state=torch.clone(self.state)
        b.legal_moves=self.legal_moves
        b.end=self.end
        return b
    
    def is_end(self) -> int: # 1 for win, 0 for draw, -1 for continue
        horizontal_sums = self.state.sum(1)
        if torch.max(torch.abs(horizontal_sums))==3:
            return 1
        vertical_sums = self.state.sum(0)
        if torch.max(torch.abs(vertical_sums))==3:
            return 1
        forward_slash = self.state[torch.arange(3), torch.arange(3)]
        backward_slash = self.state[torch.arange(3), [2,1,0]]
        if abs(torch.sum(forward_slash)) == 3 or abs(torch.sum(backward_slash))==3:
            return 1
        if len((self.state==0).nonzero()) == 0: # all squares are filled
            return 0
        return -1
    

class DeepQModel(nn.Module): #inputs the current state of the board, outputs the expected return for making move at each grid
    def __init__(self):
        super(DeepQModel, self).__init__()
        self.l1 = nn.Linear(9, 1000, dtype=float)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(1000, 80, dtype=float)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(80, 80, dtype=float)
        self.a3 = nn.ReLU()
        self.lo = nn.Linear(80,9, dtype=float)
        self.ao = nn.Softplus()

        self.optimiser = optim.Adam(self.parameters(),lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma=0.8
        self.verbose = False
    def forward(self, mat3x3: torch.Tensor):
        x = torch.clone(mat3x3.flatten())
        # x+=1
        # print(f"processed x is:\n{x}")
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        # x = self.a3(self.l3(x))
        x = self.ao(self.lo(x))
        return x.reshape(3,3)
    

class Agent:
    def __init__(self, id: int, model: DeepQModel):
        self.model = model
        self.id = id # id can not be zero

    def infer(self, board: Board, epsilon: float = 0.0):
        self.model.eval()
        if random() < epsilon:
            Q = torch.randperm(9)
            Q=Q.reshape(3,3).to(device)
        else:
            with torch.no_grad():
                Q = self.model(board.state)
        Q+= (board.legal_moves==0)*100
        min_a = torch.min(Q)
        AM =(Q==min_a) # AM is short for "addition matrix"
        n_moves = len(AM[AM])
        if n_moves != 1:
            ord=torch.randperm(n_moves).to(device)
            Q[AM]+= ord
            AM = (Q==min_a)
        return AM*self.id

    def play(self):
        end = False
        board = Board()
        model = self.model
        model.eval()
        with torch.no_grad():
            if random()<0.5:
                while not end and not board.end:
                    raw =model(board.state)
                    bot_move = self.infer(board)
                    print(f"bot made move \n{bot_move}\n based on Q \n{ raw+(board.legal_moves==0)*100}")
                    board.write(bot_move)
                    print(board.state)
                    move = input("")
                    a,b = move.split(",")
                    AM = torch.zeros((3,3)).to(device)
                    AM[int(a),int(b)] = -1
                    board.write(AM)
            else:
                while not end and not board.end:
                    print(board.state)
                    move = input("")
                    a,b = move.split(",")
                    AM = torch.zeros((3,3)).to(device)
                    AM[int(a),int(b)] = -1
                    board.write(AM)
                    raw =model(board.state)
                    bot_move = self.infer(board)
                    print(f"bot made move \n{bot_move}\n based on Q \n{ raw+(board.legal_moves==0)*100}")
                    board.write(bot_move)

                
                
class Train:
    def __init__(self):
        self.model = DeepQModel().to(device)
        self.a1 = Agent(1, self.model)
        self.a2 = Agent(1, self.model)

        self.r_win = 0
        self.r_lose = 20
        self.r_draw = 1
        self.r_move = 1

        self.epsilon = 1.1
        self.gamma = 0.8
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(),lr=0.0005)

        self.verbose = False

    def backprop(self, model: DeepQModel, s: Board, AM: torch.Tensor, value):
        model.train()
        states= generate_symmetries(s.state)
        positions=generate_symmetries(AM)
        loss =0
        for state, position in zip(states,positions):
            loss+= self.backprop_ind(model,state,position,value)
        return loss

    def backprop_ind(self, model: DeepQModel, s: torch.Tensor, AM: torch.Tensor, value):
        self.optimiser.zero_grad()
        Q = model(s)
        label = torch.clone(Q)
        label[AM!=0]=value
        loss= self.criterion(Q, label)
        loss.backward()
        self.optimiser.step()
        if self.verbose:
            print(f"s: \n{s}")
            print(f"Q: \n{Q}")
            print(f"label: \n{label}")
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
                if s2.winner == 0: # draw
                    tr_p += self.r_draw
                    tr_o += self.r_draw
                    running_loss+=self.backprop(player.model,s0, AM0, self.r_draw)
                    running_loss+=self.backprop(opp.model,s1, AM1, self.r_draw)
                else:
                    tr_p += self.r_lose
                    tr_o += self.r_win
                    running_loss+=self.backprop(player.model, s0, AM0, self.r_lose)
                    running_loss+=self.backprop(opp.model,s1, AM1, self.r_win)
            else:
                s2s = generate_symmetries(s2.state)
                s0s = generate_symmetries(s0.state)
                Qs = [player.model(s).detach() for s in s2s]
                AM0s = generate_symmetries(AM0)
                tr_p += self.r_move
                for Q, AM,s in zip(Qs, AM0,s0s):
                    running_loss+=self.backprop_ind(player.model,s, AM, self.r_move+self.gamma*Q[AM!=0])
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
        episodes = 1000,
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

def generate_symmetries(mat3x3: torch.Tensor) -> list[torch.Tensor]:
    results = []
    results.append(torch.clone(mat3x3))
    results.append(torch.flip(mat3x3,[1,0]))
    results.append(torch.fliplr(mat3x3))
    results.append(torch.flipud(mat3x3))
    results.append(torch.transpose(mat3x3,1,0))
    return results


    
play=train_loop().a1.play


                    

 
# begin
# make actual move at s0 using Q0, creating s1

# while not end:
# opponent makes theoretical best move/random move at s1 using Q1, creating s2
    # if board ends with draw, update model with 
    #   [draw] at s0
    # else:
    #   [loss] at s0
    #   [win] at s1
# inference to get Q2 at s2
# update model at s0 using Q2

# s1 becomes s0, s2 becomes s1
# Q1 becomes Q0, Q2 becomes Q1


# components:
# 1. Board
# 2. Reward giving environment
# 3. Agent which uses Q values to make moves and update model
# 4. Model(s) which gives Q values