import numpy as np
from random import random
from tack_ultils import coords_to_AM

AMNone = np.array([
    -2, 2, 0,
    2,0,-2,
    0,-2,2,
])

class Board:
    def __init__(self,  init_state: np.ndarray | None=None, isclone=False):
        if not isclone:
            if init_state is None:
                self.state = np.zeros((3,3))
            else:
                self.state = init_state
            self.end =False
            self.winner = None
            self.depth = 0

    def write(self, addition_matrix: np.ndarray):
        self.state=addition_matrix+self.state
        value = addition_matrix[addition_matrix!=0][0] # assumes one move at a time
        match self.is_end():
            case 1:
                self.end = True
                self.winner = int(value)
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
        b.state=np.copy(self.state)
        b.end=self.end
        b.depth = self.depth
        return b
    
    def is_end(self) -> int: # 1 for win, 0 for draw, -1 for continue
        horizontal_sums = self.state.sum(1)
        if np.max(np.abs(horizontal_sums))==3:
            return 1
        vertical_sums = self.state.sum(0)
        if np.max(np.abs(vertical_sums))==3:
            return 1
        forward_slash = self.state[np.arange(3), np.arange(3)]
        backward_slash = self.state[np.arange(3), [2,1,0]]
        if abs(np.sum(forward_slash)) == 3 or abs(np.sum(backward_slash))==3:
            return 1
        if len(self.state[self.state==0]) == 0: # all squares are filled
            return 0
        return -1

    def __repr__(self):
        return str(self.state)
    
    def __str__(self):
        return self.__repr__()

class ReplayBuffer:
    # all internal representations of type np.ndarray
    def __init__(self, board: Board):
        self.states: list[np.ndarray] = [extract_state(board)]
        self.actions: list[np.ndarray] = [AMNone]
        self.depth = 0
    
    def __len__(self):
        return len(self.states)
    
    def __str__(self):
        result = ""
        for state, action in zip(self.states, self.actions):
            result += f"{state.reshape(3,3)}\n"
            if not np.equal(action, AMNone):
                result += f"{action.reshape(3,3)}n"
        return result
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]

    def append(self, board: Board, new_action: np.ndarray):
        self.states.append(extract_state(board))
        self.actions[-1] = new_action.flatten()
        self.actions.append(AMNone)
        self.depth+=1

    def empty(self):
        self.states = self.states[-1:]
        self.actions = self.actions[-1:]

    
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

def extract_state(board: Board) -> np.ndarray:
    return board.state.flatten()

def generate_symmetries(matflat: np.ndarray) -> np.ndarray:
    opps = [
        np.copy,
        np.flip,
        np.fliplr,
        np.flipud,
        np.transpose,
        np.rot90,
        lambda x: np.rot90(x, 3),
    ]
    if np.array_equal(matflat, AMNone):
        return AMNone.reshape(1,-1).repeat(len(opps), 0)
    else:
        mat3x3 = matflat.reshape(3,3)
        resulT3x3 = np.stack([f(mat3x3) for f in opps])
        return resulT3x3.reshape(-1, 9)