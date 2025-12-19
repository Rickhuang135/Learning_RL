import torch
import torch.nn as nn
import torch.optim as optim

from tack_board import *
from tack_nn import A2CModel
from tack_ultils import find_model
from tack_ultils import is_end
from global_vars import *
from datetime import datetime
import time
import json
import numpy as np

from tack_benchmark import benchmark

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)
torch.serialization.add_safe_globals([A2CModel])

MODELPATH = "./tack_models/"

class EpisodeData:
    def __init__(self):
        self.Pi_loss = torch.zeros(1, device=device)
        self.V_loss = torch.zeros(1, device=device)
        self.entropy_loss = torch.zeros(1, device=device)

    def __repr__(self):
        return f"Pi_loss: {self.Pi_loss.item():.5f} \t V_loss: {self.V_loss.item():.5f} \t Entropy_loss {self.entropy_loss.item():.5g} "

    def __str__(self):
        return self.__repr__()

                
class Train:
    def __init__(self, hyper_params: dict):
        self.model = A2CModel().to(device)

        self.r_win = 1
        self.r_draw = 0

        self.hyper_params = hyper_params
        self.gamma = hyper_params['gamma']
        self.entropy_beta = hyper_params['entropy_beta']
        self.criterionV = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(),lr=hyper_params['learn_rate'])
        self.replay_buffer_length = hyper_params['replay_length']
        if "model_prefix" in hyper_params.keys():
            model_paths, version = find_model(MODELPATH,hyper_params["model_prefix"])
            self.version = version
            for model_path in model_paths:
                if "model" in model_path:
                    print(f"loading model {model_path} of version {self.version}")
                    self.model=torch.load(MODELPATH+model_path,map_location=device, weights_only=False)
        self.steps = 0
        self.verbose = False

    def infer(self, states_flat: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(torch.tensor(states_flat, dtype=torch.float32).to(device), Pi_only = True)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim = -1)
            cum_dist = prob.cumsum(-1)
            idx = torch.searchsorted(cum_dist, torch.rand(states_flat.shape[0], device=device).reshape(-1,1))
        return idx.detach().cpu().numpy().flatten()
    
    def backprop_with_symmetries(self, model: A2CModel, rb: ReplayBuffer, episode_loss: EpisodeData | None = None):
        # retrieve data from replay buffer
        states, game_ids, is_first_player_turns, actions, rewards = rb.get_all() 
        augmented_states = symmetry_generator.rotate(states) # augment with symmetries added in new dimension
        unique_ids, counts = np.unique(game_ids, return_counts=True) # check for discrete games in data
        
        # Sending data to GPU
        game_ids_torch = torch.tensor(game_ids, dtype=torch.int32).to(device)
        player_ids = torch.tensor((is_first_player_turns-0.5)*2).to(device)
        augmented_actions = torch.tensor(symmetry_generator.rotate_indicies(actions)).to(device) # The start of each row is the original action
        rewards_torch = torch.tensor(rewards, dtype=torch.int8).to(device)

        # infering with model
        model.train()
        self.optimiser.zero_grad()
        logits, values = model(torch.Tensor(augmented_states.reshape(-1,9)).to(device)) # logits are flat
        values: torch.Tensor = values.reshape((-1, symmetry_generator.n_ops))
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).reshape((-1, symmetry_generator.n_ops, 9))
        prob = torch.nn.functional.softmax(logits, dim=-1).reshape((-1, symmetry_generator.n_ops, 9))

        # initialise losses
        Vloss = torch.zeros(1, device=device)
        Piloss = torch.zeros(1, device=device)
        for id, count in zip(unique_ids, counts):
            if count==1:
                continue
            # tensors on gpu
            cvalues = values[game_ids_torch == id]
            clog_prob = log_prob[game_ids_torch == id]
            cactions = augmented_actions[game_ids_torch == id]
            cplayer_ids = player_ids[game_ids_torch == id]
            crewards = rewards_torch[game_ids_torch == id]

            # calculate V loss
            cvalue_means = torch.mean(cvalues, dim=-1)
            cvalue_labels = cvalue_means[1:].clone().detach() * self.gamma
            if not (crewards==no_r).all(): # game has ended
                cvalue_labels[-1] = crewards[crewards!=no_r] * self.gamma
            cVloss = self.criterionV(cvalue_means[:-1], cvalue_labels)
            Vloss+=cVloss
            # print(cvalues)
            # print(cvalue_means)
            # print(cvalue_labels)
            # print(cVloss)

            # calculate policy loss
            advantage = (cvalue_labels - cvalue_means[:-1].detach())
            cvalid_actions = cactions[:-1,:]
            cvalid_log_prob = clog_prob[:-1,:]
            cvalid_log_prob = cvalid_log_prob.gather(dim=2, index=cvalid_actions.unsqueeze(-1)).squeeze(-1)
            grad_Pi = cvalid_log_prob.mean(dim=1)*advantage*cplayer_ids[:-1]
            cPiloss = torch.sum(grad_Pi)
            Piloss+=cPiloss

        # entropy normalisation
        entropy = -(prob*log_prob).sum()
        entropy_loss = -1 * self.entropy_beta * entropy
        (Vloss+Piloss+entropy_loss).backward()
        self.optimiser.step()

        if episode_loss is not None:
            episode_loss.Pi_loss+=torch.abs(Piloss)
            episode_loss.V_loss+=torch.abs(Vloss)
            episode_loss.entropy_loss+=entropy_loss
        
        self.steps += 1

    
    def save(self, increment_version = True, extra_info: dict = {}):
        if 'model_prefix' in self.hyper_params.keys():
            if increment_version:
                self.version[1]+=1
            prefix = self.hyper_params['model_prefix']
            version_str = '_'.join(str(x) for x in (self.version))
            saved_paths = [
                f"./{MODELPATH}/{prefix}_model#{version_str}.pt",
                f"./{MODELPATH}/{prefix}#{version_str}.json",
            ]
        else:
            now = datetime.now().strftime('%m_%d_%H%M')
            saved_paths = [
                f"./{MODELPATH}/{now}_tack3_model#0_0.pt",
                f"./{MODELPATH}/{now}_tack3#0_0.json",
            ]
        torch.save(self.model, saved_paths[0])
        info_dict = {
            'device': str(device)
        }
        info_dict.update(self.hyper_params)
        info_dict.update({ # type:ignore
            'steps': self.steps,  # type:ignore
        })
        info_dict.update(extra_info)
        with open(saved_paths[1], "w") as file:
            json.dump(info_dict, file, indent=4)
        return saved_paths

    def benchmark_handler(self, board: Board, id:int):
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(torch.tensor((board.state.flatten()), dtype=torch.float32).to(device), Pi_only = True)
            prob: torch.Tensor = torch.nn.functional.softmax(raw_output, dim=0)
            cum_dist = prob.cumsum(0)
            idx = torch.searchsorted(cum_dist, torch.rand(1, device=device))
            AM: torch.Tensor = torch.zeros_like(prob)
            AM[idx]=1
            return AM.detach().cpu().numpy().reshape((3,3))*id
    
def train_loop(
        episodes = 10000,
):  
    hyper_params = {
        'gamma':0.80,
        'entropy_beta':0.03,
        'learn_rate': 0.0001,
        'replay_length': 4,
        'parallel_games': 10,
        # 'model_prefix': '12_12_1245_tack3',
    }
    train = Train(hyper_params)
    print_period = min(episodes//10, 50)
    replay_length = hyper_params["replay_length"]
    parallel_games = hyper_params["parallel_games"]
    game_uid = parallel_games
    states = new_states = np.zeros((parallel_games,9)) # array of flat board states
    game_ids = np.arange(parallel_games, dtype=np.int32)
    is_first_player_turn = np.ones(parallel_games, dtype = np.bool) # flat array of 1s
    rewards = None
    rb = ReplayBuffer(states, game_ids, is_first_player_turn)
    begin_time = time.time()
    while game_uid-parallel_games < episodes:
        loss = EpisodeData()
        for i in range(replay_length):
            move_inds = train.infer(states) # get moves from inference
            new_states = np.copy(states) # get new states from moves
            new_states[np.arange(parallel_games),move_inds] = (is_first_player_turn-0.5)*2
            for i, state in enumerate(new_states.reshape(-1,3,3)): # for each new state:
                end = is_end(state)
                if end!=-1:
                    if rewards is None:
                        rewards = np.zeros(parallel_games, dtype=np.int8)+no_r
                    rewards[i] = (is_first_player_turn[i]-0.5)*2 if end==1 else 0 # player who just moved wins
                    game_uid += 1
                    game_ids[i] = game_uid
                    new_states[i] = np.zeros(9) # replace with empty board
            is_first_player_turn= np.invert(is_first_player_turn)
            if rewards is not None:
                is_first_player_turn[rewards!=no_r] = True
            rb.append(new_states=new_states, game_ids=game_ids, is_first_player_turn= is_first_player_turn, new_actions=move_inds, rewards= rewards)
            rewards = None
            states = new_states
        # print(rb.to_df().head(10))
        # raise Exception("stop for a moment")
        train.backprop_with_symmetries(train.model, rb, loss)
        rb.empty()
        if game_uid % print_period == 0:
            print(f"{round(game_uid/episodes*100)}% {loss}")


    time_elapsed = time.time()-begin_time
    steps_per_second = train.steps/time_elapsed
    print(f"{train.steps} steps of {hyper_params["replay_length"]*7} completed in {time_elapsed:.3f} seconds at {steps_per_second:.3f} steps/second")

    benchmark(train.benchmark_handler)

    if episodes >= 20000: # save the model
        print(f"Model saved to {train.save(increment_version=True, extra_info={
            "episodes": episodes,
            "time_elapsed": time_elapsed, 
            "steps/second": steps_per_second,
            })}")

    return train

train_res=train_loop()

# test_positions = torch.tensor([
#     [1,-1,0,
#     0,0,0,
#     0,0,0],

#     [1,-1,0,
#     0,1,0,
#     0,0,0],

#     [1,-1,0,
#     0,1,0,
#     0,0,-1],

#     [1,-1,0,
#     0,1,0,
#     1,0,-1],

#     [1,-1,0,
#     -1,1,0,
#     1,0,-1],
# ], device=device, dtype=torch.float32)

# logits, values = train_res.model(test_positions)
# probs = torch.nn.functional.softmax(logits, -1)
# from tack_ultils import pt
# for prob, value, position in zip(probs, values, test_positions):
#     pt(position)
#     pt(prob)
#     print(value)
# play(lambda board, id: train_res.a1(board) * id)

# components:
# 1. Board
# 2. Reward giving environment
# 3. Agent which uses Va values to make moves and update V
# 4. Model(s) which gives Va values

# value system:
# while not s0 end:
    # make move at s0, creating s1
    # if s1 is end state
    #   update V1 with draw or win
    # 
    # update V0 with s0
    # update Pi with V0
    # s0 = s1