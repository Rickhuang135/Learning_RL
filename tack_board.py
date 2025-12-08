import torch
import numpy as np

from random import random
from tack_ultils import generate_symmetries
from tack_ultils import coords_to_AM
from device import device

class Board:
    def __init__(self,  init_state=None, isclone=False):
        if not isclone:
            if init_state is None:
                self.state =torch.tensor(np.zeros((3,3))).to(device)
            else:
                self.state: torch.Tensor = init_state
            self.legal_moves=torch.where(self.state==0,1,0)
            self.end =False
            self.winner = None
            self.depth = 0

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
        # result.state*=-1
        result.depth +=1
        return result

    def copy(self):
        b=Board(isclone=True)
        b.state=torch.clone(self.state)
        b.legal_moves=torch.clone(self.legal_moves)
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
