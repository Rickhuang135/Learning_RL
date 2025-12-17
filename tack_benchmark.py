from tack_board import Board
from tack_abprune import prune
import numpy as np
import json
from os.path import exists
from random import choice

class perfectMinMax:
    stored_file_name = "./perfectMinMax.json"

    def __init__(self):
        if exists(self.stored_file_name):
            with open(self.stored_file_name, 'r') as file:
                self.dict=json.load(file)
        else:
            print("no existing tree found, generating one")
            full_tree= prune(Board(), id=1, perfect_min_max=True).to_dict()
            with open(self.stored_file_name, 'w') as file:
                json.dump(full_tree, file, indent=4)
            self.dict = full_tree

    def infer(self, board: Board, id=1):
        state = board.state.flatten()
        first_player_moves = np.where(state==1)[0]
        second_layer_moves = np.where(state==-1)[0]
        longer = len(first_player_moves)>len(second_layer_moves)
        if longer:
            second_layer_moves = np.append(second_layer_moves, 0)
        all_moves = np.stack([first_player_moves,second_layer_moves]).transpose().flatten()
        if longer:
            all_moves = all_moves[:-1]
        c = self.dict
        for move in all_moves:
            c=c["children"][c["actions"].index(move)]
        actions = []
        best_value = 100 * id
        for i, child in enumerate(c["children"]):
            value = child["value"]
            if (id > 0 and value < best_value) or (id < 0 and value > best_value):
                best_value = value
                actions = [c["actions"][i]]
            elif value == best_value:
                actions.append(c["actions"][i])
        AM = np.zeros(9)
        AM[choice(actions)] = id
        return AM.reshape(3,3)
        

def benchmark(benchmarked, benchmarker = perfectMinMax().infer, n_runs = 200):
    n_draws = 0
    n_wins = 0
    n_loss = 0
    for _ in range(n_runs//2):
        state = Board()
        mover = (benchmarker, 1)
        while not state.end:
            infer, id = mover
            state=state.next(infer(state, id=id))
            if infer==benchmarker:
                mover = (benchmarked, -1)
            else:
                mover = (benchmarker, 1)
        match state.winner:
            case 1: n_loss+=1
            case 0: n_draws+=1
            case -1: n_wins+=1
    for _ in range(n_runs//2):
        state = Board()
        mover = (benchmarked, 1)
        while not state.end:
            infer, id = mover
            state=state.next(infer(state, id=id))
            if infer==benchmarker:
                mover = (benchmarked, 1)
            else:
                mover = (benchmarker, -1)
        match state.winner:
            case -1: n_loss+=1
            case 0: n_draws+=1
            case 1: n_wins+=1
    print(f"{n_wins}/{n_draws}/{n_loss} out of {n_runs} total")
