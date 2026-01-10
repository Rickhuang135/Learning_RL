import torch
from global_vars import *
from hypr_prams import hyper_params
from replayBuffer import ReplayBuffer
from ultils import is_end
from train import *
import time
import numpy as np

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision= 3)


def train_loop(
        steps = 100_000,
        record = False,
        hyper_params = hyper_params,
):  
    train = Train(hyper_params)
    print_period = max(min(steps//50, 100),1) # 1 ≤ print_period ≤ 100
    write_loss_period = max(min(steps//5, 200), 1) # 1 ≤ write_loss_period ≤ 200
    print(f"print period at {print_period}")
    replay_length = hyper_params["replay_length"]
    parallel_games = hyper_params["parallel_games"]
    game_uid = parallel_games
    if test_arr1 is None:
        starting_pos = np.zeros(9)
        states = new_states = np.zeros((parallel_games,9)) # array of flat board states
    else:
        starting_pos = np.array(test_arr1)
        states = new_states = np.repeat(np.expand_dims(starting_pos, 0), parallel_games, axis=0)
    game_ids = np.arange(parallel_games, dtype=np.int32)
    is_first_player_turn = np.ones(parallel_games, dtype = np.bool) # flat array of 1s
    rewards = None
    rb = ReplayBuffer(states, game_ids, is_first_player_turn)
    lc = LossCollector()
    begin_time = time.time()
    for step in range(1, steps+1):
        for _ in range(replay_length-1):
            move_inds = train.infer(states) # get moves from inference
            new_states = np.copy(states) # get new states from moves
            new_states[np.arange(parallel_games),move_inds] = (is_first_player_turn-0.5)*2
            for i, state in enumerate(new_states.reshape(-1,3,3)): # for each new state:
                end = is_end(state)
                if end!=-1:
                    if rewards is None:
                        rewards = np.zeros(parallel_games, dtype=np.int8)+r_none
                    rewards[i] = r_win*(is_first_player_turn[i]-0.5)*2 if end==1 else r_draw # player who just moved wins
                    game_ids[i] = game_uid
                    game_uid += 1
                    new_states[i] = starting_pos.copy() # replace with empty board
            is_first_player_turn= np.invert(is_first_player_turn)
            if rewards is not None:
                is_first_player_turn[rewards!=r_none] = True
            rb.append(new_states=new_states, game_ids=game_ids, is_first_player_turn= is_first_player_turn, new_actions=move_inds, rewards= rewards)
            rewards = None
            states = new_states
        train.backprop_with_symmetries(train.model, rb, lc)
        if step % print_period == 0:
            progress_str = f"{(step*100)//steps}%"
            print(f"{progress_str:<4}{lc.last_loss()}")

        if step % write_loss_period == 0 and record:
            lc.write()
            lc.empty()

        rb.empty()
        if not record:
            lc.empty()


    time_elapsed = time.time()-begin_time
    steps_per_second = train.steps/time_elapsed
    batch_size = (hyper_params["replay_length"]-1)*symmetry_generator.n_ops*parallel_games
    print(f"{game_uid - parallel_games} episodes completed")
    print(f"{batch_size} values per batch")
    print(f"{train.steps} steps completed in {time_elapsed:.3f} seconds at {steps_per_second:.3f} steps/second")


    if steps >= 10000: # save the model
        print(f"Model saved to {train.save(increment_version=True, extra_info={
            "episodes": game_uid-parallel_games,
            "time_elapsed": time_elapsed, 
            "steps/second": steps_per_second,
            "batch_size": batch_size
            })}")

    return train

if __name__ == '__main__':
    from benchmark import benchmark
    train_res=train_loop(
        steps = 0,
        record=True,
    )
    benchmark(train_res.benchmark_handler)
    from play import play
    play(train_res.benchmark_handler)

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