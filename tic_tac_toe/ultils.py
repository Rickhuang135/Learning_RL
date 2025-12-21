from global_vars import *
import numpy as np
import os

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
        np.copy,
        np.flip,
        np.fliplr,
        np.flipud,
        np.transpose,
        np.rot90,
        lambda x: np.rot90(x, 3),
    ]):
        self.n_ops = len(operations)
        inds = np.arange(9).reshape((3,3))
        self.new_inds = np.concat([f(inds) for f in operations]).reshape(self.n_ops,9)
        self.augmented_inverses = np.append(np.argsort(self.new_inds, axis=1), np.zeros((self.n_ops,1), dtype=np.int8)+self.noIndex, axis=1) # augmented with noIndex entry

    def rotate_single(self, mat1d: np.ndarray) -> np.ndarray:
        return mat1d[self.new_inds]
    
    def rotate_indicies(self, indices: np.ndarray) -> np.ndarray:
        valid_indices = indices.copy()
        valid_indices[indices==no_action] = self.noIndex
        new_indicies = self.augmented_inverses[:, valid_indices].T
        new_indicies[new_indicies==self.noIndex] = no_action
        return new_indicies
    
    def rotate(self, matnd: np.ndarray) -> np.ndarray:
        return matnd[:, self.new_inds]

symmetry_generator = SymmetryGenerator()