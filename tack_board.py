import torch

from random import random
from tack_ultils import coords_to_AM
from device import device

mat3x3_template = torch.zeros(3,3, dtype=torch.float32, device= device)
AMNone = torch.tensor([
    -2, 2, 0,
    2,0,-2,
    0,-2,2,
], dtype=torch.float32, device= device)

class Board:
    def __init__(self,  init_state=None, isclone=False):
        if not isclone:
            if init_state is None:
                self.state = torch.clone(mat3x3_template)
            else:
                self.state: torch.Tensor = init_state
            self.end =False
            self.winner = None
            self.depth = 0

    def write(self, addition_matrix):
        self.state+=addition_matrix
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
        result.depth +=1
        return result

    def copy(self):
        b=Board(isclone=True)
        b.state=torch.clone(self.state)
        b.end=self.end
        b.depth = self.depth
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
    
    def __repr__(self):
        return str(self.state.cpu().numpy())
    
    def __str__(self):
        return self.__repr__()
    
    def generate_symmetries(self) -> list:
        mat3x3 = self.state
        results =[Board(init_state=mat) for mat in generate_symmetries(mat3x3)]
        for result in results: 
            result.depth=self.depth
            if self.end:
                result.end = self.end
                result.winner=self.winner
            
        return results
    
class ReplayBuffer:
    # all internal representations of type torch.Tensor
    def __init__(self, board: Board):
        self.states: list[torch.Tensor] = [extract_state(board)]
        self.actions: list[torch.Tensor] = [AMNone]
        self.depth = 0
    
    def __len__(self):
        return len(self.states)
    
    def __str__(self):
        result = ""
        for state, action in zip(self.states, self.actions):
            result += f"{state.reshape(3,3).cpu().numpy()}\n"
            if not torch.equal(action, AMNone):
                result += f"{action.reshape(3,3).cpu().numpy()}\n"
        return result
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]

    def append(self, board: Board, new_action: torch.Tensor):
        self.states.append(extract_state(board))
        self.actions[-1] = new_action.flatten()
        self.actions.append(AMNone)
        self.depth+=1

    def empty(self):
        self.states = self.states[-1:]
        self.actions = self.actions[-1:]

class BoardDataset(torch.utils.data.Dataset):
    def __init__(self, rb: ReplayBuffer):
        self.states = torch.concat([generate_symmetries(s) for s in rb.states], dim=0)
        self.actions = torch.concat([generate_symmetries(a) for a in rb.actions], dim=0)
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, ind):
        return self.states[ind], self.actions[ind]

    
def play(Agent,init_board=None,player_turn=False, player_id=-1):
    if init_board is None:
        board=Board()
        if random()<0.4:
            player_turn=True
            player_id = 1
        return play(Agent, board, player_turn, player_id)
    else:
        board: Board = init_board
        print(board)
        if board.end:
            print(f"{board.winner} has won!")
            return None
        if player_turn:
            move = input("Enter move (x,y): ")
            a,b = move.split(",")
            AM = coords_to_AM((int(a), int(b))) * player_id
        else:
            AM = Agent(board, id=player_id*-1)
        board.write(AM)
        return play(Agent, board, not player_turn, player_id)

def extract_state(board: Board) -> torch.Tensor:
    return board.state.flatten()

def generate_symmetries(matflat: torch.Tensor) -> torch.Tensor:
    opps = [
        torch.clone,
        lambda x: torch.flip(x, [1,0]),
        torch.fliplr,
        torch.flipud,
        lambda x: torch.transpose(x, 1, 0),
        torch.rot90,
        lambda x: torch.rot90(x, 3),
    ]
    if torch.equal(matflat, AMNone):
        return AMNone.repeat(len(opps), 1)
    else:
        mat3x3 = matflat.reshape(3,3)
        resulT3x3 = torch.stack([f(mat3x3) for f in opps])
        return torch.flatten(resulT3x3, start_dim=1, end_dim=2)