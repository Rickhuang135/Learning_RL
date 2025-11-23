import torch
import numpy as np
from random import choice

class MCTS:
    c = np.sqrt(2) # exploration parameter

    def __init__(self, action, head):
        self.head:MCTS =head
        self.action = action
        self.visits=1
        self.children: list[MCTS]=[]
        # implement code to get reward
        self.reward: int|None = None

    def UCB(self):
        if self.reward is None:
            raise Exception("Node not finished, cannot evaluate")
        return self.reward + self.c*np.sqrt(np.log(self.head.visits)/self.visits)

    def next(self, actions: torch.Tensor):
        new_acts = []
        existing_acts=[child.action for child in self.children]
        for act in actions:
            if not act in existing_acts:
                new_acts.append(act)
        if len(new_acts):
            return MCTS(choice(new_acts), self)
        UCBs= np.array([child.UCB() for child in self.children])
        return self.children[np.argmin(UCBs)]

    # run from starting state to end
        # at starting state, gets list of actions
        # check for new action, take if exist
        # check which child has lowest UCB
        # repeat until new action is reached 
    # when creating new leaf
        # randomly make moves until terminal state reach
        # create new child with reward and visit count
    # create leaf based on first action and reward

# def train_loop2():
#     s0 = Board()
#     player = Agent(1, DeepQModel())
#     head = MCTS(0, None)
#     actions = torch.arange(9)[s0.legal_moves.flatten()==1]
#     current_node = head.next(actions)
#     while not current_node.reward is None:
#         current_node=current_node.next(s0)
#     total_reward = 0
#     while not s0.end:
#         s0.write(player.infer(s0,2))
#         total_reward+=1
#         if s0.end:
#             current_node.reward=20
        
