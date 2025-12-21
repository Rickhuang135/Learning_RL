import numpy as np
from global_vars import *
import pandas as pd
from os import path

class ReplayBuffer:
    # all internal representations of type np.ndarray
    def __init__(self, states: np.ndarray, game_ids: np.ndarray, is_first_player_turn: np.ndarray, rewards: np.ndarray | None = None):
        self.n_parallel = states.shape[0]
        self.MNone = - np.ones(self.n_parallel, dtype=np.int8)

        self.states: list[np.ndarray] = [states]
        self.game_ids = [np.copy(game_ids)]
        self.is_first_player_turn = [is_first_player_turn]
        self.actions: list[np.ndarray] = [self.MNone]
        self.rewards = []
        if rewards is None:
            self.default_r()
        else:
            self.rewards.append(rewards)
        self.offset = 0
    
    def __len__(self):
        return len(self.states)

    @staticmethod 
    def join(array: list):
        res = np.stack(array)
        if len(res.shape) == 3: # array has shape (replay_length, n_parallel, 9)
            return res.transpose((1,0,2)) # new shape (n_parallel, replay_length, 9)
        else:
            return res.transpose((1,0)) # new shape (n_parallel, replay_length)
    
    def get_all(self, join_method = None):
        if join_method is None:
            join_method = self.join

        all_states = join_method(self.states[self.offset:]) # (n_parallel, replay_length, 9)
        all_game_ids = join_method(self.game_ids[self.offset:]) # (n_parallel, replay_length)
        all_is_first_player_turns = join_method(self.is_first_player_turn[self.offset:]) # (n_parallel, replay_length)
        all_actions = join_method(self.actions[self.offset:]) # (n_parallel, replay_length)
        all_rewards = join_method(self.rewards[self.offset:]) # (n_parallel, replay_length)
        return all_states, all_game_ids, all_is_first_player_turns, all_actions, all_rewards

    def __str__(self):
        result = ""
        for state, action in zip(self.states, self.actions):
            result += f"{state.reshape(3,3)}\n"
            if not np.equal(action, self.MNone):
                result += f"{action.reshape(3,3)}n"
        return result
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]

    def append(self, new_states: np.ndarray, game_ids:np.ndarray, is_first_player_turn: np.ndarray, new_actions: np.ndarray, rewards: np.ndarray | None = None):
        self.states.append(new_states)
        self.game_ids.append(np.copy(game_ids))
        self.is_first_player_turn.append(is_first_player_turn)
        self.actions[-1] = new_actions
        self.actions.append(self.MNone)
        if rewards is not None:
            self.rewards[-1] = rewards
        self.default_r()

    def next(self):
        self.offset = len(self) - 1
    
    def empty(self):
        self.offset=0
        self.states = self.states[-1:]
        self.game_ids = self.game_ids[-1:]
        self.is_first_player_turn = self.is_first_player_turn[-1:]
        self.actions = self.actions[-1:]
        self.rewards = self.rewards[-1:]

    def default_r(self): # add 3 to indicate no reward
        self.rewards.append(np.zeros(self.n_parallel, dtype=np.int8)+r_none)
    
    def to_df(self, offset: int | None = None) -> pd.DataFrame:
        og_offset = self.offset
        if offset is not None:
            self.offset = offset
        states, game_ids, is_first_player_turns, actions, rewards = self.get_all(np.concat)
        self.offset = og_offset

        df = pd.DataFrame(
            states,
        )
        df.index.name = "idx"
        df["game_id"] = game_ids
        df["turn"] = is_first_player_turns
        df["action"] = actions
        df["rewards"] = rewards
        
        return df

    def write_csv(self) -> None:
        location = REPLAYPATH
        if path.isfile(location): # file already exists
            df = self.to_df(offset=1)
            df.to_csv(location, mode='a', index=False, header = False)
        else: # file doesn't exists
            df = self.to_df(offset=0)
            df.to_csv(location, mode='w', index=False, header = True)