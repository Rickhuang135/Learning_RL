from global_vars import *
import pandas as pd
import numpy as np
from ultils import symmetry_generator

def load_replay(path = REPLAYPATH):
    df = pd.read_csv(path)
    df.index.name = 'idx'
    return df.sort_values(by=["game_id", "idx"])

class GameSingle:
    board_indexes= [str(x) for x in range(9)]
    def __init__(self, data: pd.DataFrame):
        self.data=data
        extracted_boards = data[self.board_indexes].to_numpy().reshape(-1,3,3,2).transpose(3,0,1,2)
        self.states = extracted_boards[0]
        self.probs = extracted_boards[1]

    def __repr__(self):
        result = f"{self.data[["turn","action","rewards","policy_gradients","value_predictions","value_labels","Pi_loss","V_loss","entropy_loss"]]}"
        return f"{result}\n{self.states}\n{self.probs}"
        

class Data:
    def __init__(self):
        self.replay_buffer = load_replay(REPLAYPATH)
        self.loss_collector = load_replay(LOSSPATH)
        self.shape = self.replay_buffer.shape

    def get(self, game_id: int, symmetry: int ) -> GameSingle:
        replay = self.replay_buffer[self.replay_buffer["game_id"] == game_id ].reset_index(drop=True)
        loss = self.loss_collector[self.loss_collector["game_id"] == game_id ]
        loss = loss.iloc[symmetry::symmetry_generator.n_ops].reset_index(drop=True)
        return GameSingle(pd.concat([replay,loss], axis= 1).drop("game_id", axis=1))

a = Data()
g = a.get(3,0)
print(g)