import torch
import torch.nn as nn
from tack_board import Board
NEG_INF = -1e20

class DeepQModel(nn.Module): #inputs the current state of the board, outputs the expected return for making move at each grid
    def __init__(self):
        super(DeepQModel, self).__init__()
        self.l1 = nn.Linear(9, 1000, dtype=torch.float32)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(1000, 80, dtype=torch.float32)
        self.a2 = nn.ReLU()
        self.lo = nn.Linear(80,9, dtype=torch.float32)
        self.ao = nn.Softplus()

    def forward(self, mat3x3: torch.Tensor) -> torch.Tensor:
        x = torch.clone(mat3x3.flatten())
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.ao(self.lo(x))
        return x
    

class PolicyModel(nn.Module): #inputs the current state of the board, outputs the probability of making each move
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.l1 = nn.Linear(9,1000, dtype=torch.float32)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(1000,100, dtype=torch.float32)
        self.a2 = nn.ReLU()
        self.lo = nn.Linear(100,9, dtype=torch.float32)
    
    def forward(self, board: Board) -> torch.Tensor:
        x = torch.clone(board.state.flatten())
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.lo(x)
        mask = torch.clone(board.legal_moves.flatten())
        neg_inf_tensor = torch.full_like(x, NEG_INF)
        x = torch.where(mask==1, x, neg_inf_tensor)
        return x

class ValueModel(nn.Module): #inputs the current state of the board, outputs the expected result from this position
    def __init__(self):
        super(ValueModel, self).__init__()
        self.l1 = nn.Linear(9, 1000, dtype=torch.float32)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(1000, 160, dtype=torch.float32)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(160, 9, dtype=torch.float32)
        self.a3 = nn.ReLU()
        self.lo = nn.Linear(9,1, dtype=torch.float32)

    def forward(self, mat3x3: torch.Tensor) -> torch.Tensor:
        x = torch.clone(mat3x3.flatten())
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.a3(self.l3(x))
        x = self.lo(x)
        return x
    