import torch
import torch.nn as nn

class DeepQModel(nn.Module): #inputs the current state of the board, outputs the expected return for making move at each grid
    def __init__(self):
        super(DeepQModel, self).__init__()
        self.l1 = nn.Linear(9, 1000, dtype=float)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(1000, 80, dtype=float)
        self.a2 = nn.ReLU()
        self.lo = nn.Linear(80,9, dtype=float)
        self.ao = nn.Softplus()

        self.gamma=0.8
        self.verbose = False
    def forward(self, mat3x3: torch.Tensor) -> torch.Tensor:
        x = torch.clone(mat3x3.flatten())
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.ao(self.lo(x))
        return x
    

class PolicyModel(nn.Module): #inputs the current state of the board, outputs the probability of making each move
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.l1 = nn.Linear(9,100, dtype=float)
        self.a1 = nn.LeakyReLU()
        self.l2 = nn.Linear(100,100, dtype=float)
        self.a2 = nn.Softplus()
        self.lo = nn.Linear(100,9, dtype=float)
        self.ao = nn.Softmax(dim=0)
    
    def forward(self, mat3x3: torch.Tensor) -> torch.Tensor:
        x = torch.clone(mat3x3.flatten())
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.lo(x)
        # print(x)
        x = self.ao(x)
        # x = self.ao(self.lo(x))
        return x