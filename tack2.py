import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from random import randint
from random import random
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
win_count = 0

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
    
    def is_legal(self, point: tuple) -> bool:
        if self.state[point[0],point[1]] != 0:
            return False
        return True

    def move(self, point: tuple, player:int) -> int:
        if not self.is_legal(point):
            self.end = True
            return 20
        self.state[point[0],point[1]]= player
    
        is_end = self.is_end()
        match is_end:
            case -1: # not ended
                return 1
            case 0: # draw
                self.end = True
                return 1
            case 1: # mover has won
                self.end = True
                return 0
            case _:
                raise Exception(f"unrecognised ending state '{is_end}'")

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
        self.l3 = nn.Linear(256, 256, dtype=float)
        self.a3 = nn.ReLU()
        self.lo = nn.Linear(256,9, dtype=float)
        self.ao = nn.ReLU()

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        # x = self.a3(self.l3(x))
        x = self.ao(self.lo(x))
        return x

def i2p(index) -> tuple:
    if not isinstance(index, int):
        raise Exception(f"invalid index {index}")
    return index//3, index%3

def min_a(Q: torch.Tensor) -> tuple:
    index = torch.argmin(Q).item()
    return i2p(index), index

def rand_a() -> tuple:
    index = randint(0,8)
    return i2p(index), index

def get_other_move(game: TickTackToe, model:DeepQModel):
    # aquire Qs for moves of player
    oQ = model(game()*-1)
    o_min_a, o_min_a_ind = min_a(oQ)
    # use best move if legal
    if game.is_legal(o_min_a):
        o_a = o_min_a
    else:
        matching_indices = (game.state == 0).nonzero(as_tuple=False)
        o_a = tuple(matching_indices[randint(0, len(matching_indices)-1)])
    return o_a

state = TickTackToe()
model = DeepQModel().to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(),lr=0.001)
player = 1
other = -1
gamma = 0.8
epsilon = 1.0
episodes = 1000
steps = 0

interval = episodes//10
step_size = epsilon/10
if interval==0:
    interval=1

def back(s0: torch.Tensor, p_reward, p_min_a1, v=False):
    optimiser.zero_grad()
    pQ0 = model(s0)
    label = torch.clone(pQ0)
    label[p_a0_ind] = p_reward + gamma*p_min_a1
    loss = criterion(pQ0, label)
    loss.backward()
    if v:
        print(f"s0: \n{s0.reshape(3,3)}")
        print(f"pQ0: \n{pQ0.reshape(3,3)}")
        print(f"label: \n{label.reshape(3,3)}")
        print(f"loss: {loss}")
    optimiser.step()
    global steps
    steps+=1
    return loss.item()

optimiser.zero_grad()
with torch.no_grad():
    pQ0 = model(state())
s0 = torch.clone(state())
begin = True
start_time = time()
for i in range(episodes):
    runningloss = 0
    total_reward = 0
    while not state.end:
        p_min_a1 = 0
        with torch.no_grad():
            # choose best move or random move
            if random() < epsilon:
                p_a0, p_a0_ind = rand_a() # random move
            else:
                p_a0, p_a0_ind = min_a(pQ0) # best move
            # make move and get reward
            p_reward = state.move(p_a0, player)
            if p_reward == 0:
                win_count+=1
            if not state.end: # player has not made illegal move or won
                o_a = get_other_move(state, model)
                o_reward = state.move(o_a, other)
                if state.end: # other has won or drew
                    if o_reward == 0: # other has won
                        p_reward = 10
                    else: # other has drawn
                        p_reward = 2
                else:
                    # get Q of best move for current player
                    pQ1 = model(state())
                    p_min_a1 = torch.min(pQ1).item()
        # update neural net work with reward + gamma * Q min a
        v = False
        if episodes-i <= 1:
            v=True
        runningloss+=back(s0, p_reward, p_min_a1, v=v)
        total_reward+=p_reward
        # change variables for next loop
        del pQ0
        del s0
        s0 = torch.clone(state())
        pQ0 = pQ1 #type: ignore
    print(f"epsilon: {epsilon:1f} \t runningloss during episode: {runningloss:.5g} \t total_reward: {total_reward}")
    state.reset()
    begin = not begin
    if begin:
        with torch.no_grad():
            state.move(get_other_move(state,model), other)
    s0 = torch.clone(state())
    if i % interval ==0:
        epsilon -= step_size

time_elapsed = time()-start_time
steps_per_second = steps/time_elapsed
print(f"win_count == {win_count} at {steps_per_second:.3f} steps/second")

def play():
    end = False
    game=TickTackToe()
    with torch.no_grad():
        if random()<0.5:
            while not end and not game.end:
                bQs = model(game())
                bot_move = i2p(int(torch.argmin(bQs)))
                print(f" based on: \n{bQs.reshape(3,3)}")
                print(f"bot made move {bot_move} and got reward {game.move(bot_move,1)}")
                print(game.state)
                move = input("")
                a,b = move.split(",")
                game.move((int(a),int(b)),-1)
        else:
            while not end and not game.end:
                print(game.state)
                move = input("")
                a,b = move.split(",")
                game.move((int(a),int(b)),-1)
                bQs = model(game())
                bot_move = i2p(int(torch.argmin(bQs)))
                print(f" based on: \n{bQs.reshape(3,3)}")
                print(f"bot made move {bot_move} and got reward {game.move(bot_move,1)}")