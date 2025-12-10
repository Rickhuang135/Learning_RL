import torch
from device import device
import os

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

def find_model(folder_path: str, model_prefix: str, version: list|None = None) -> tuple[list, list]:
    all_models = os.listdir(f"./{folder_path}")
    target_model_filter = filter(lambda x: x.startswith(model_prefix), all_models)
    if version is None:
        target_model = list(target_model_filter)
        version = [-1 for _ in range(4)]
        version_str = ''
        for model in target_model:
            v_start = model.rfind("#")+1
            v_end = model.rfind(".")
            c_version_str = model[v_start:v_end] # version format x_x
            version_info = [int(x) for x in c_version_str.split("_")]
            if version_info[0]>=version[0] and version_info[1]>version[1]:
                version = version_info
                version_str = c_version_str
        target_model = list(filter(lambda x: f"#{version_str}." in x, target_model))   
    else:
        version_str = '_'.join(version)
        target_model = list(filter(lambda x: version_str in x, target_model_filter))
    if len(target_model)==0:
        raise Exception(f"Model {model_prefix} not found in {folder_path} ")
    return target_model, version