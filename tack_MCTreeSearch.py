import torch
import numpy as np
import math
import time
from tack_board import Board

class MCTS:
    c = math.sqrt(2) # exploration parameter

    def __init__(self, actions: torch.Tensor, head = None, terminal = False):
        self.head:MCTS | None = head
        if head is not None:
            head.append(self)
        self.actions = actions # use indexes i/8
        self.terminal = terminal
        self.n_terminal_children = 0
        self.children: list[MCTS | None] = [None for _ in actions]
        self.visits = 0
        self.optimal = None
        self.expanding_action= None
        if head is None or not head.minimising:
            self.minimising = True
        else:
            self.minimising = False
        self.value= 0
        self.mean_value = 0

    def get_root(self):
        if self.head is None:
            return self
        else:
            return self.head
    
    def expand(self, id0: int, chain: list|None=None):
        if chain is None:
            chain = []
        action_ind, child = self.pick_action_child()
        chain.append(action_ind)
        if child is None:
            self.expanding_action = action_ind
            depth = len(chain)
            ids=torch.arange(depth)
            chain_values = torch.ones(depth)
            chain_values[ids%2==0] *= id0
            chain_values[ids%2!=0] *= id0*-1
            AM = torch.zeros(9)
            AM[torch.stack(chain, dim=0)] = chain_values
            return self, AM.reshape(3,3), depth
        
        if child.terminal:
            child.simulated_visit
            return self.get_root().expand(id0)
        
        return child.expand(id0, chain)
        

    def simulate_visit(self):
        self.percolate_up(self.optimal, False)

    def __str__(self):
        child_values=  [(None if child is None else child.mean_value) for child in self.children]
        result = f"MCTS {'minimising' if self.minimising else 'maximising'} node with value {self.value}, {'terminal' if self.terminal else ''} \nwith actions {self.actions} \nand children {child_values}"
        return result
    
    def __repr__(self):
        return self.__str__()

    def UCB(self):
        if self.head is None:
            raise Exception("Root of tree, UCB cannot be calculated")
        if self.value is None:
            raise Exception("Node has no defined value, cannot evaluate")
        if self.minimising:
            sign = -1
        else:
            sign = 1
        return self.mean_value + sign* self.c*math.sqrt(math.log(self.head.visits)/self.visits)
    

    def pick_action_child(self) -> tuple: #recursively finds next action to explore
        if self.terminal:
            raise Exception(f"Terminal state doesn't need to be explored")
        next_action = None
        next_child = None
        if self.minimising:
            Best_UCB = 100
        else:
            Best_UCB = -100
        for child, action in zip(self.children, self.actions):
            if child is None:
                return action, None
            elif child.terminal:
                continue
            else:
                UCBi = child.UCB()
                if (self.minimising and UCBi < Best_UCB) or (not self.minimising and UCBi > Best_UCB):
                    next_child = child
                    next_action = action
                    Best_UCB = UCBi
        return next_action, next_child

    def append(self, child):
        self.children[torch.where(self.actions==self.expanding_action)[0]] = child
    
    def percolate_up(self, value, terminal=False) -> None:
        self.value+=value
        self.visits+=1
        self.mean_value = self.value / self.visits

        if terminal:
            self.n_terminal_children+=1
        if self.n_terminal_children == len(self.children):
            self.terminal = True

        if self.terminal:
            if len(self.children)==0 or None in self.children:
                self.optimal = self.value
            else:
                child_values = [child.optimal for child in self.children] # type: ignore
                if self.minimising:
                    self.optimal = min(child_values) # type: ignore
                else:
                    self.optimal = max(child_values) # type: ignore

        if self.head is not None:
            self.head.percolate_up(value, self.terminal)

    # run from starting state to end
        # at starting state, gets list of actions
        # check for new action, take if exist
        # check which child has lowest UCB
        # repeat until new action is reached 
    # when creating new leaf
        # randomly make moves until terminal state reach
        # create new child with value and visit count
    # create leaf based on first action and value

def ind_to_AM(index_out_of_8):
    result = torch.zeros(9)
    result[index_out_of_8] = 1
    return result.reshape(3,3)

def actions_from_board(board: Board):
    legal_moves = board.legal_moves.flatten()
    return torch.where(legal_moves==1)[0]

def random_move(board: Board, id=1):
    actions = actions_from_board(board)
    # print(actions)
    ind = actions[torch.randint(low=0, high=actions.numel(), size=(1,))]
    return ind_to_AM(ind) * id

def search(
        board: Board,
        id = 1,
        run_time = 1.000, # in seconds
        n_runs = 2000
):
    start_time = time.time()
    current_run = 0
    head = MCTS(actions_from_board(board))
    # begin looping according to time constraint
    while not head.terminal and (time.time()-start_time < run_time) and current_run<n_runs:
        current_run+=1
        current_board = board.copy()
        parent, AM_multiple, depth = head.expand(id)
        c_id = id*((-1)**(depth+1))
        current_board.write(AM_multiple)
        new_node = MCTS(actions_from_board(current_board), head=parent)
        b_depth = depth
        while not current_board.end: # finish episode to get value
            c_id *= -1
            depth += 1
            current_board = current_board.next(random_move(current_board, id=c_id))
        
        if b_depth == depth:
            new_node.terminal = True

        if current_board.winner == 0: # draw
            value = 0
        elif current_board.winner == id: # won
            value = -(10-depth)
        else: # lost
            value = (10-depth)
        new_node.percolate_up(value)
    print(f"{current_run} runs completed")
    return head

def infer(s: Board, id=1):
    root=search(s, id=id)
    print(root)
    child_values = [ (None if c is None else c.optimal) for c in root.children]
    print(child_values)
    # return p.action[child_values.index(min(child_values))].reshape(3,3)

s0 = Board()
# s0.write(torch.Tensor([
#     [0,0,0],
#     [0,0,0],
#     [0,0,0],
# ]))

infer(s0, id=1)