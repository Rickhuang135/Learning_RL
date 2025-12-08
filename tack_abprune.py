from tack_board import *
import torch
from device import device

class P:
    def __init__(self, value=None, depth=0):
        self.action:list = []
        self.children: list[P] = []
        self.value =value
        self.depth=depth
    def append(self, actions, P):
        self.action.append(actions)
        self.children.append(P)

    def __repr__(self):
        res = f"{self.value}"
        # children = [f"{c.depth*'-->'} action {a} to {c}" for a,c in zip(self.action,self.children) ]
        children = [f"\n{'-->'*c.depth}{c}" for c in self.children ]
        return res + "".join(children)


def prune(
        s: Board, 
        id = 1,
        min_node=True, 
        alpha=-100, # maximum achievable value
        beta=100, # minimum achievable value
        depth=0):
    # assumes initial node is minimising
    if s.end:
        if s.winner==0:
            return P(0, depth=depth)
        elif min_node: # lost
            return P(10-depth, depth=depth)
        else: # won
            return P(-10+depth, depth=depth)
    else:
        # if depth>=9:
        #     print(depth)
        #     print(s)
        #     print(s.legal_moves)
        #     print(s.end)
        #     raise Exception("WTF")
        node = P(depth=depth)
        legal = s.legal_moves.flatten().tolist()
        for index,m in enumerate(legal):
            if m:
                AM = torch.zeros(9).to(device)
                AM[index]=id
                sn = s.next(AM.reshape(3,3))
                if min_node: # beta is local value, alpha is upstream value
                    res = prune(sn, id*-1, False, alpha, beta, depth+1)
                    beta = min(res.value,beta)
                else: # is max_node, alpha is local value, beta is upstream value
                    res = prune(sn, id*-1, True, alpha, beta, depth+1)
                    alpha = max(res.value,alpha)
                node.append(AM, res)
                if beta<=alpha: # prune when, case min-node: upstream value is greator than current
                    break       #             case max-node: upstream value is lessor than current
        if min_node:
            node.value=beta
        else:
            node.value=alpha
        return node

def infer(s: Board, id=1):
    p=prune(s, id=id)
    child_values = [c.value for c in p.children]
    print(child_values)
    return p.action[child_values.index(min(child_values))].reshape(3,3)

def value(s: Board):
    p=prune(s)
    child_values = [c.value for c in p.children]
    return min(child_values)

play(infer)