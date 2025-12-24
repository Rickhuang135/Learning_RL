from global_vars import *
import numpy as np
import torch
import os
from device import device

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
        version_str = '_'.join([str(x) for x in version])
        target_model = list(filter(lambda x: version_str in x, target_model_filter))
    if len(target_model)==0:
        raise Exception(f"Model {model_prefix} not found in {folder_path} ")
    return target_model, version

def is_end(state: np.ndarray) -> int: # 1 for win, 0 for draw, -1 for continue
    horizontal_sums = state.sum(1)
    if np.max(np.abs(horizontal_sums))==3:
        return 1
    vertical_sums = state.sum(0)
    if np.max(np.abs(vertical_sums))==3:
        return 1
    forward_slash = state[np.arange(3), np.arange(3)]
    backward_slash = state[np.arange(3), [2,1,0]]
    if abs(np.sum(forward_slash)) == 3 or abs(np.sum(backward_slash))==3:
        return 1
    if len(state[state==0]) == 0: # all squares are filled
        return 0
    return -1


class SymmetryGenerator:
    noIndex = 9

    def __init__(self, operations: list = [
        torch.clone,
        # lambda x: torch.flip(x, [1,0]),
        # torch.fliplr,
        # torch.flipud,
        # lambda x: torch.transpose(x, 1, 0),
        # torch.rot90,
        # lambda x: torch.rot90(x, 3),
    ]):
        self.n_ops = len(operations)
        inds = torch.arange(9, dtype=torch.int32, device=device).reshape((3,3))
        self.new_inds = torch.concat([f(inds) for f in operations]).reshape(self.n_ops,9)
        self.augmented_inverses = torch.cat([torch.argsort(self.new_inds, dim=1), torch.zeros((self.n_ops,1), dtype=torch.int32, device=device)+self.noIndex], dim=1) # augmented with noIndex entry

    def rotate_single(self, mat1d: torch.Tensor) -> torch.Tensor:
        return mat1d[self.new_inds]
    
    def rotate_indicies(self, indices: torch.Tensor) -> torch.Tensor:
        valid_indices = indices.clone()
        valid_indices[indices==no_action] = self.noIndex
        new_indicies = self.augmented_inverses[:, valid_indices].transpose(0,1)
        new_indicies[new_indicies==self.noIndex] = no_action
        return new_indicies
    
    def rotate(self, matnd: torch.Tensor) -> torch.Tensor:
        return matnd[:, self.new_inds]

symmetry_generator = SymmetryGenerator()

# state = torch.tensor([
#     [
#         1,0,0,
#         0,0,0,
#         0,0,0
#     ],
#     [
#         1,0,0,
#         -1,0,0,
#         0,0,0
#     ],
#     [
#         1,0,1,
#         -1,0,0,
#         0,0,0   
#     ]
# ], device=device)
# print(symmetry_generator.rotate(state).reshape(-1,3,3))

# test symmetries