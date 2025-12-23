from global_vars import *
from os import path
import pandas as pd
import numpy as np
import json

np.set_printoptions(suppress=True, precision=3)

def print_slices_horizontal(arr, sep="   "):
    slices = []

    for i in range(arr.shape[0]):
        lines = np.array2string(arr[i], separator=' ').splitlines()

        # Remove outer brackets
        lines[0] = " "+lines[0][1:]
        lines[-1] = lines[-1][:-1]

        slices.append(lines)

    # Print row-wise
    for r in range(len(slices[0])):
        print(sep.join(s[r] for s in slices))

class npDict(dict[str, np.ndarray]):
    def __init__(self, name_arrays: dict[str, np.ndarray] | list[tuple[str, np.ndarray]]):
        dict.__init__(self, name_arrays)
    
    def __str__(self):
        result =""
        for name, array in self.items():
            result+=f"{name}: {array}\n"
        return result

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self)

class GameSingle(npDict):

    def __init__(self, name_arrays: dict[str, np.ndarray]| list[tuple[str, np.ndarray]]):
        npDict.__init__(self, name_arrays)
        self.standard_shape = self["game_id"].shape

    def print(self):
        print(self.filter_name("loss", True).to_df())
        print(self.filter_shape().to_df())
        # print(self["state"])
        state_shapes = self.filter_shape((self.standard_shape[0], 3,3))
        # Convert each slice to strings
        for name, state_shape in state_shapes.items():
            print(name)
            print_slices_horizontal(state_shape)
  

    def filter_name(
            self,
            term= "loss", 
            inclusion = False # include or exclude term
        ) -> npDict:

        result = []
        for name, array in self.items():
            if (inclusion and (term in name)) or not (inclusion or (term in name)):
                result.append((name,array))
        return npDict(result)
    
    def filter_shape(self, shape: None | tuple = None) -> npDict:
        if shape is None:
            shape = self.standard_shape
        result = []
        for name, item in self.items():
            if item.shape == shape:
                result.append((name, item))
        return npDict(result)

    def print_alternate(self) -> None:
        sequential = list(self.filter_name().items())
        length = sequential[0][1].shape[0]
        for i in range(length):
            for name, array in sequential:
                print(name)
                print(array[i])
                print()


class Data:
    @staticmethod
    def dtype_from_str(string: str):
        match string:
            case "<class 'numpy.int32'>": return np.int32
            case "<class 'numpy.float32'>": return np.float32
            case _: raise Exception(f"unexpected dtype {string}, can not be converted")

    def __init__(self):
        metadata_path = f"{REPLAYPATH}metadata.json"
        if path.isfile(metadata_path):
            with open(metadata_path) as file:
                metadata = json.load(file)
        else:
            raise Exception(f"No metadata file found at {metadata_path}")
        steps = metadata["steps"]
        name_maps: list[tuple[str,np.memmap]] = []
        for item in metadata["items"]:
            shape: list = item["shape"]
            name: str = item["name"]
            shape[0] = steps
            mm = np.memmap(
                f"{REPLAYPATH}{name}.dat",
                dtype = self.dtype_from_str(item["dtype"]),
                mode = "r",
                shape = tuple(shape)
            )
            name_maps.append((name, mm))
        self.data: dict[str, np.memmap] = dict(name_maps)
        self.episodes = np.max(self.data["game_id"])
        self.symmetries = self.data["game_id"].shape[-1]

    def all(self) -> dict[str, np.ndarray]:
        result = []
        for name, arr in self.data.items():
            if len(arr.shape) == 4 : # (steps, n_parallel, rb_length, n_symmetries)
                result.append((name, arr.reshape(-1, arr.shape[1], arr.shape[3]))) # (steps × rb_length, n_parallel, n_symmetries)
            elif len(arr.shape) == 5: # (steps, n_parallel, rb_length, n_symmetries, 9)
                result.append((name, arr.reshape(-1, arr.shape[1], arr.shape[3], 9))) # (steps × rb_length, n_parallel, n_symmetries, 9)
        return dict(result)
    
    def get(self, game_id: int, symmetry: int) -> GameSingle:
        d_nRB = self.data
        id_mask_lg = (d_nRB["game_id"][:,:,:,symmetry] == game_id) # (steps, n_parallel, rb_length, n_symmetries)
        id_mask_sm = np.any(id_mask_lg, axis=(1,2))
        reduced_name_maps = []
        for name, arr in d_nRB.items():
            if len(arr.shape)==1:
                result = arr[id_mask_sm]
            else:
                result = arr[:,:,:,symmetry][id_mask_lg]
            if arr.shape[-1] == 9:
                result = result.reshape(-1, 3, 3)
            reduced_name_maps.append((name, result))
        return GameSingle(reduced_name_maps)
    
    def sample(self, game_id: int | None = None, symmetry: int |None = None):
        return self.get(
            np.random.randint(0, self.episodes) if game_id is None else game_id, 
            np.random.randint(0, self.symmetries) if symmetry is None else symmetry,
            )
    

a = Data()
g = a.sample(None, 0)
g.print()