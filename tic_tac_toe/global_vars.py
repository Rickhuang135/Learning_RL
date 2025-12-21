# keep this file to plain python
MODELPATH = "./nn_models/"
REPLAYPATH = "./replay/rb.csv"
LOSSPATH = "./replay/lc.csv"
r_draw = 0

# state value representing win, multiply by player_id
#   for any r ∈ ℝ, r > r_draw
r_win = 10

# default value representing no reward given
#   for any r ∈ ℝ, |r| > r_win
r_none = 13

# default value representing no action taken
#   for any a ∈ ℤ, a < 0, a > 8
no_action = -1

# test_arr1 = [
#     1,-1,0,
#     0,0,0,
#     0,0,0,
# ]
test_arr1 = None