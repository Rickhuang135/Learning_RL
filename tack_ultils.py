import torch

def generate_symmetries(mat3x3: torch.Tensor) -> list[torch.Tensor]:
    results = []
    results.append(torch.clone(mat3x3))
    results.append(torch.flip(mat3x3,[1,0]))
    results.append(torch.fliplr(mat3x3))
    results.append(torch.flipud(mat3x3))
    results.append(torch.transpose(mat3x3,1,0))
    return results

def pt(matflat: torch.Tensor):
    mat3x3=matflat.reshape(3,3)
    print(torch.round(mat3x3.detach().cpu(), decimals=3).numpy())

def normalise_Pi(flat_raw_dist: torch.Tensor, legal_moves_flat: torch.Tensor):
    raw_dist = torch.clone(flat_raw_dist)
    raw_dist[legal_moves_flat!=1] = 0
    return raw_dist/torch.sum(raw_dist)