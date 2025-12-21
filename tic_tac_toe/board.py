from global_vars import *
import numpy as np
from random import random
from ultils import is_end
import pandas as pd

if  test_arr1 is None:
    starting_pos = np.zeros((3,3))
else:
    starting_pos = np.array(test_arr1)

class Board:
    uid = 0

    def __init__(self,  init_state: np.ndarray | None=None, isclone=False):
        if not isclone:
            if init_state is None:
                self.state = starting_pos.reshape((3,3))
            else:
                self.state = init_state
            self.end =False
            self.winner = None
            self.depth = 0
            self.id = Board.uid
            Board.uid += 1

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
        b.id = self.id
        return b
    
    def is_end(self) -> int: # 1 for win, 0 for draw, -1 for continue
        return is_end(self.state)

    def __repr__(self):
        return str(self.state)
    
    def __str__(self):
        return self.__repr__()

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
            AM = np.zeros((3,3))
            AM[int(a), int(b)] = player_id
        else:
            AM = Agent(board, id=player_id*-1)
        board.write(AM)
        return play(Agent, board, not player_turn, player_id)