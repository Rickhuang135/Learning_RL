from tack_board import Board
from tack_ultils import coords_to_AM
import torch

class Game:
    def __init__(self, state_list: list[Board] | None = None):
        self.traverse = -1
        if state_list is None:
            self.states: list[Board] = []
        else:
            self.states = state_list
    
    def append(self, state: Board):
        self.states.append(state)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.traverse+=1
        if self.traverse >= len(self):
            self.traverse = -1
            raise StopIteration
        else:
            return self.states[self.traverse]
        
