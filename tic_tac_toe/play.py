from board import Board
from random import random
import numpy as np

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
            AM = np.zeros((3,3))
            AM[int(a), int(b)] = player_id
        else:
            AM = Agent(board, id=player_id*-1)
        board.write(AM)
        return play(Agent, board, not player_turn, player_id)