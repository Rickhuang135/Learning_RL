import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from random import randint
from random import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TickTackToe:
    def __init__(self, state: torch.Tensor|None=None):
        shape = (3,3)
        if state is None:
            self.state = torch.tensor(np.zeros(shape)).to(device)
        else:
            self.state = state
        self.shape = shape
        self.end = False

    def __call__(self):
        return self.state.flatten()

    def is_end(self) -> bool: # only check if the game has ended not who has won
        horizontal_sums = self.state.sum(1)
        if torch.max(torch.abs(horizontal_sums))==3:
            return True
        vertical_sums = self.state.sum(0)
        if torch.max(torch.abs(vertical_sums))==3:
            return True
        forward_slash = self.state[torch.arange(3), torch.arange(3)]
        backward_slash = self.state[torch.arange(3), [2,1,0]]
        if abs(torch.sum(forward_slash)) == 3 or abs(torch.sum(backward_slash))==3:
            return True
        return False
    
    def is_legal(self, point: tuple) -> bool:
        if self.state[point[0],point[1]] != 0:
            return False
        return True

    def move(self, point: tuple, player:int) -> int:
        if not self.is_legal(point):
            self.end = True
            return -20
        self.state[point[0],point[1]]= player
        if self.is_end():
            self.end = True
            return 0
        else:
            return -1

    def reset(self):
        self.state*=0
        self.end = False


class DeepQModel(nn.Module): #inputs the current state of the board, outputs the expected return for making move at each grid
    def __init__(self):
        super(DeepQModel, self).__init__()
        self.l1 = nn.Linear(9, 256, dtype=float)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256, dtype=float)
        self.a2 = nn.ReLU()
        self.lo = nn.Linear(256,9, dtype=float)
        self.ao = nn.ReLU()

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.ao(self.lo(x))
        return x

def i2p(index) -> tuple:
    if not isinstance(index, int):
        raise Exception(f"invalid index {index}")
    return index//3, index%3

def max_a(Q: torch.Tensor) -> tuple:
    index = torch.argmax(Q).item()
    return i2p(index), index

def rand_a() -> tuple:
    index = randint(0,8)
    return i2p(index), index


state = TickTackToe()
model = DeepQModel().to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(),lr=0.001)
player = 1
other = -1
gamma = 0.8
epsilon = 0.9
episodes = 10
interval = episodes//10
step_size = epsilon/10

def back(s0, p_reward, p_max_a1):
    optimiser.zero_grad()
    pQ0 = model(s0)
    label = torch.clone(pQ0)
    label[p_a0_ind] = p_reward + gamma*p_max_a1
    print(f"s0: {s0}")
    print(f"pQ0: {pQ0}")
    print(f"label: {label}")
    loss = criterion(pQ0, label)
    loss.backward()
    optimiser.step()
    return loss.item()

optimiser.zero_grad()
with torch.no_grad():
    pQ0 = model(state())
s0 = state()
for i in range(episodes):
    runningloss = 0
    while not state.end:
        p_max_a1 = 0
        with torch.no_grad():
            # choose best move or random move
            if random() < epsilon:
                p_a0, p_a0_ind = rand_a() # random move
            else:
                p_a0, p_a0_ind = max_a(pQ0) # best move
            # make move and get reward
            p_reward = state.move(p_a0, player)
            if not state.end: # player has not made illegal move or won
                # aquire Qs for moves of other player
                oQ = model(state() * -1)
                o_max_a, o_max_a_ind = max_a(oQ)
                # other player makes best legal move and get reward
                if state.is_legal(o_max_a):
                    o_a = o_max_a
                else:
                    matching_indices = (state.state == 0).nonzero(as_tuple=False)
                    o_a = tuple(matching_indices[0])
                o_reward = state.move(o_a, other)
                if state.end: # other has won
                    p_reward=-10
                else:
                    # get Q of best move for current player
                    pQ1 = model(state())
                    p_max_a1 = torch.max(pQ1).item()
        # update neural net work with reward + gamma * Q max a
        runningloss+=back(s0, p_reward, p_max_a1)
        # change variables for next loop
        del pQ0
        del s0
        s0 = state()
        pQ0 = pQ1 #type: ignore
    print(runningloss)
    state.reset()
    if i % interval ==0:
        epsilon -= step_size
        print(epsilon)

print(state.state)

def play():
    end = False
    game=TickTackToe()
    while not end and not game.end:
        print(game.move(max_a(model(game()))[0],1))
        print(game.state)
        move = input("")
        a,b = move.split(",")
        game.move((int(a),int(b)),-1)
        