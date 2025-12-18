import numpy as np
from random import random
from tack_ultils import coords_to_AM
from tack_ultils import is_end
import pandas as pd


class Board:
    uid = 0

    def __init__(self,  init_state: np.ndarray | None=None, isclone=False):
        if not isclone:
            if init_state is None:
                self.state = np.zeros((3,3))
            else:
                self.state = init_state
            self.end =False
            self.winner = None
            self.depth = 0
            self.id = Board.uid
            Board.uid += 1

    def write(self, addition_matrix: np.ndarray):
        self.state=addition_matrix+self.state
        value = addition_matrix[addition_matrix!=0][0] # assumes one move at a time
        match self.is_end():
            case 1:
                self.end = True
                self.winner = int(value)
            case 0:
                self.end = True
                self.winner = 0
            case -1:
                self.end = False
        
    def next(self, addition_matrix):
        result = self.copy()
        result.write(addition_matrix)
        result.depth +=1
        return result

    def copy(self):
        b=Board(isclone=True)
        b.state=np.copy(self.state)
        b.end=self.end
        b.depth = self.depth
        b.id = self.id
        return b
    
    def is_end(self) -> int: # 1 for win, 0 for draw, -1 for continue
        return is_end(self.state)

    def __repr__(self):
        return str(self.state)
    
    def __str__(self):
        return self.__repr__()

class ReplayBuffer:
    # all internal representations of type np.ndarray
    def __init__(self, states: np.ndarray, game_ids: np.ndarray, is_first_player_turn: np.ndarray, rewards: np.ndarray | None = None):
        self.n_parallel = states.shape[0]
        self.MNone = - np.ones(self.n_parallel)

        self.states: list[np.ndarray] = [states]
        self.game_ids = [np.copy(game_ids)]
        self.is_first_player_turn = [is_first_player_turn]
        self.actions: list[np.ndarray] = [self.MNone]
        self.rewards = []
        if rewards is None:
            self.default_r()
        else:
            self.rewards.append(rewards)
        self.depth = 0
    
    def __len__(self):
        return len(self.states)
    
    def to_df(self):
        all_game_ids = np.concat(self.game_ids)
        all_states = np.concat(self.states)
        all_is_first_player_turns = np.concat(self.is_first_player_turn)
        all_actions = np.concat(self.actions)
        all_rewards = np.concat(self.rewards)


        df = pd.DataFrame(
            all_states,
        )
        df.index.name = "idx"
            # columns=["state","game_id","action","is_first_player_turn","reward"]
        df["game_id"] = all_game_ids
        df["player_turn"] = all_is_first_player_turns
        df["action"] = all_actions
        df["rewards"] = all_rewards
        return df.sort_values(by=["game_id", "idx"])

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
        self.depth+=1

    def empty(self):
        self.states = self.states[-1:]
        self.game_ids = self.game_ids[-1:]
        self.is_first_player_turn = self.is_first_player_turn[-1:]
        self.actions = self.actions[-1:]
        self.rewards = self.rewards[-1:]

    def default_r(self): # add 3 to indicate no reward
        self.rewards.append(np.zeros(self.n_parallel, dtype=np.int8)+3)
    
def play(Agent,init_board=None,player_turn=False, player_id=-1):
    if init_board is None:
        board=Board()
        if random()<0.4:
            player_turn=True
            player_id = 1
        return play(Agent, board, player_turn, player_id)
    else:
        board: Board = init_board
        print(board)
        if board.end:
            print(f"{board.winner} has won!")
            return None
        if player_turn:
            move = input("Enter move (x,y): ")
            a,b = move.split(",")
            AM = coords_to_AM((int(a), int(b))) * player_id
        else:
            AM = Agent(board, id=player_id*-1)
        board.write(AM)
        return play(Agent, board, not player_turn, player_id)

class SymmetryGenerator:
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

    def rotate_single(self, mat1d: np.ndarray):
        return mat1d[self.new_inds]
    
    def rotate_indicies(self, indices: np.ndarray):
        return np.argsort(self.new_inds, axis=1)[:, indices].T
    
    def rotate(self, matnd: np.ndarray):
        return matnd[:, self.new_inds].reshape(-1, 9)


def generate_symmetries(mat1d: np.ndarray) -> np.ndarray:
    opps = [
        np.copy,
        np.flip,
        np.fliplr,
        np.flipud,
        np.transpose,
        np.rot90,
        lambda x: np.rot90(x, 3),
    ]
    mat3x3 = mat1d.reshape(3,3)
    resulT3x3 = np.stack([f(mat3x3) for f in opps])
    return resulT3x3.reshape(-1, 9)