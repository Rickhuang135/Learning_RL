import numpy as np
import math
import time
from board import Board

class MCTS:
    c = math.sqrt(2) # exploration parameter

    def __init__(self, actions: np.ndarray = np.zeros(0), leaf = False):
        self.head:MCTS | None = None
        if leaf:
            self.leaf = self.terminal = True
        else:
            self.leaf= self.terminal = False
        self.actions = actions # use indexes i/8
        self.n_terminal_children = 0
        self.children: list[MCTS | None] = [None for _ in actions]
        self.visits = 0
        self.optimal = None  # only exists for terminal nodes
        self.expanding_action= None
        self.minimising = True
        self.value= 0   # total value
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
            ids=np.arange(depth)
            chain_values = np.ones(depth)
            chain_values[ids%2==0] *= id0
            chain_values[ids%2!=0] *= id0*-1
            AM = np.zeros(9)
            AM[np.stack(chain, axis=0)] = chain_values
            return self, AM.reshape(3,3), depth
        
        if child.terminal:
            child.visits+=1
            self.percolate_up(child.optimal, False)
            return self.get_root().expand(id0)
        
        return child.expand(id0, chain)
      
    def __str__(self):
        if self.terminal:
            child_values = [(None if c is None else c.optimal) for c in self.children] 
        else:
            child_values=  [(None if child is None else child.mean_value) for child in self.children]
        if self.leaf:
            leaf_or_terminal_text= "leaf"
        elif self.terminal:
            leaf_or_terminal_text="terminal"
        else:
            leaf_or_terminal_text=''
        result = f"MCTS {'minimising' if self.minimising else 'maximising'} node with mean value {self.mean_value}, {leaf_or_terminal_text} \nwith actions {self.actions} \nand children {child_values}"
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
        child.head=self
        child.minimising = not self.minimising
        temp = np.where(self.actions==self.expanding_action)
        self.children[temp[0][0]] = child
        return child
    
    def percolate_up(self, value, terminal=False) -> None:
        self.value+=value
        self.visits+=1
        self.mean_value = self.value / self.visits

        if terminal:
            self.n_terminal_children+=1
            if self.n_terminal_children == len(self.children):
                self.terminal = True

        if self.terminal:
            if self.leaf:
                self.optimal = self.value
            else:
                child_values = [child.optimal for child in self.children] # type: ignore
                if self.minimising:
                    self.optimal = min(child_values) # type: ignore
                else:
                    self.optimal = max(child_values) # type: ignore

        if self.head is not None:
            self.head.percolate_up(value, self.terminal)

    def infer(self):
        best_action_ind = 0
        if self.minimising:
            best_value = 100
        else:
            best_value = -100

        for i, child in enumerate(self.children):
            if child is None: # This child node hasn't been explored
                c_value = 7
            elif child.terminal == True:
                c_value = child.optimal
            else:
                c_value = child.mean_value
            
            if self.minimising and c_value < best_value or not self.minimising and c_value > best_value: # type:ignore
                best_value = c_value
                best_action_ind = i

        return self.actions[best_action_ind]

    def to_dict(self):
        result = {
            "value": self.optimal if self.terminal else self.mean_value,
            "actions": self.actions.tolist(),
            "children": [],
        }
        for child in self.children:
            if child is not None:
                result["children"].append(child.to_dict())
            else:
                result["children"].append(None)
        return result

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
    result = np.zeros(9)
    result[index_out_of_8] = 1
    return result.reshape(3,3)

def get_actions(board: Board):
    return np.where(board.state.flatten()==0)[0]

def random_move(board: Board, id=1):
    actions = get_actions(board)
    ind = np.random.choice(actions)
    return ind_to_AM(ind) * id

def search(
        board: Board,
        id = 1,
        head: MCTS | None = None,
        run_time = 0.500, # in seconds
        n_runs = 4000
):
    start_time = time.time()
    current_run = 0
    if head is None:
        head = MCTS(get_actions(board))
    else:
        head = head
    # begin looping according to time constraint
    while not head.terminal and (time.time()-start_time < run_time) and current_run<n_runs:
        current_run+=1
        current_board = board.copy()
        parent, AM_multiple, depth = head.expand(id)
        c_id = id*((-1)**(depth))
        current_board.write(AM_multiple)

        if current_board.end:
            new_node = parent.append(MCTS(leaf=True))
        else:
            new_node = parent.append(MCTS(get_actions(current_board)))
            # print(current_board)
            while not current_board.end: # finish episode to get value
                depth += 1
                current_board = current_board.next(random_move(current_board, id=c_id))
                c_id *= -1
                # print(current_board)
                # print()
        
        if current_board.winner == 0: # draw
            value = 0
        elif c_id == id: # lost
            value = (10-depth)
        else: # won
            value = -(10-depth)
        new_node.percolate_up(value)
        # print(current_board)
        # print(f"added now node {new_node}")
    # print(f"{current_run} runs completed")
    return head

def infer(s: Board, id=1):
    root=search(s, id=id)
    # child_values = [ (None if c is None else c.optimal) for c in root.children]
    # print(root)
    # print(child_values)
    return ind_to_AM(root.infer())*id

if __name__ == "__main__":
    from benchmark import benchmark
    benchmark(infer, n_runs=20)
    from play import play
    s0 = Board()
    root=infer(s0, 1)
    play(infer)