import torch
from device import device

def generate_symmetries(mat3x3: torch.Tensor) -> list[torch.Tensor]:
    results = []
    results.append(torch.clone(mat3x3))
    results.append(torch.flip(mat3x3,[1,0]))
    results.append(torch.fliplr(mat3x3))
    results.append(torch.flipud(mat3x3))
    results.append(torch.transpose(mat3x3,1,0))
    return results

def pt(matflat: torch.Tensor):
    if len(matflat) == 9:
        mat3x3= matflat.reshape(3,3)
    else:
        mat3x3 = matflat
    print(torch.round(mat3x3.detach().cpu(), decimals=3).numpy())

def coords_to_AM(xy: tuple):
    x, y = xy
    AM = torch.zeros((3,3)).to(device)
    AM[x,y] = 1
    return AM