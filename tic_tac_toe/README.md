main.py is an A2C (advantage actor critic) algorithm for solving tic tac toe. Maximum performance of 0/934/66 achieved after 400,000 steps of training with hyper parameters:

'gamma':1,
'entropy_beta':0.62,
'learn_rate': 0.0001,
'replay_length': 4,
'parallel_games': 5,

Models trained with more than 10,000 steps are automatically saved to MODELPATH in global variables.

Pass record=True argument into train_loop function to write training logs. Access with debug.py.