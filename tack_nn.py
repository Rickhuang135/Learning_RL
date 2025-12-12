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
    
class A2CModel(nn.Module): # Advantage Actor Critic Model
    def __init__(self):
        super(A2CModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1800),
            nn.ReLU(),
            nn.Linear(1800, 1160),
            nn.ReLU(),
            nn.Linear(1160,160),
            nn.ReLU()
        )
        self.Pi = nn.Linear(160,9)
        self.V = nn.Linear(160,1)

    
    def forward(self, state: torch.Tensor, Pi_only = False, V_only = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = self.encoder(state)
        if V_only:
            return self.V(x)
        else:
            logits = self.Pi(x)
            neg_inf_tensor = torch.full_like(logits, NEG_INF)
            Pi_out = torch.where(state==0, logits, neg_inf_tensor) # state==0 yeilds boolean mask for legal moves
            if Pi_only:
                return Pi_out
            else:
                return Pi_out, self.V(x)

    