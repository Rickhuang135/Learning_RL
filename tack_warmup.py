import numpy as np
# goal, brute force calculate it

board = np.zeros((3,3)) # 0 means not filled, 1 is player 1, 2, is player 2.
# print(board)

def check_end(board_state: np.ndarray):
    rows, columns = board_state.shape
    for row in board_state:
        if row[0] == 0:
            continue
        if np.sum(row==row[0]) == columns:
            return row[0]
        
    for column in board_state.T:
        if column[0] == 0:
            continue
        if np.sum(column==column[0]) == rows:
            return column[0]
    
    if rows == columns:
        if board_state[0,0]!= 0:
            forward_slash = [board_state[x,x] for x in range(rows)]
            if np.sum(forward_slash==forward_slash[0]) == rows:
                return board_state[0,0]
        if board_state[0,rows-1]!=0:
            backward_slash = [board_state[rows-x-1,x] for x in range(rows)]
            if np.sum(backward_slash==backward_slash[0]) == rows:
                return board_state[0,rows-1]
            
    return 0
    
    

def play(board_state, side, change_board = False):
    is_end = check_end(board_state)
    if is_end>0:
        return is_end
    rows, columns = board_state.shape
    empty_spots =np.asarray(np.ravel(board_state)==0).nonzero()[0]
    if len(empty_spots) == 0:
        return 3/2
    results = []
    board_states = []
    for spot in empty_spots:
        cur_board = np.copy(board_state)
        cur_board[spot//columns, spot%columns]=side
        board_states.append(cur_board)
        results.append(play(cur_board, 3-side))

    results = np.array(results)

    if change_board:
        # [print(x) for x in board_states]
        # print(results)
        return board_states[np.abs(side-results).argmin()]
    
    chosen = results[np.abs(side-results).argmin()]
    return chosen+(1.5-chosen)*0.1


def loop():
    board = np.zeros((3,3))
    while isinstance(board, np.ndarray):
        print(board)
        position = input("input position: ")
        a,b = position.split(",")
        board[int(a),int(b)]=1
        board = play(board, 2, True)
    print(f"player {board} has won!")



test_board = np.array([
    [2, 1, 0], 
    [0, 1, 0], 
    [0, 0, 0]]
)

fail_board_1 = np.array([
    [2, 1, 0],
    [0, 1, 0],
    [0, 2, 0]
])

fail_board_2 = np.array([
    [2, 1, 0],
    [0, 1, 0],
    [2, 0, 0]
])

# print(play(test_board,2,True))
# print(play(test_board,2, True))
loop()