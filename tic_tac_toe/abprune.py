from board import Board
import numpy as np

class P:
    def __init__(self, actions: np.ndarray = np.zeros(0), value=None, depth=0):
        self.actions:np.ndarray = actions
        self.children: list[P] = []
        self.value =value
        self.depth=depth
    def append(self, P): # expects exploration in order
        self.children.append(P)

    def __repr__(self):
        res = f"{self.value}"
        # children = [f"{c.depth*'-->'} action {a} to {c}" for a,c in zip(self.actions,self.children) ]
        children = [f"\n{'-->'*c.depth}{c}" for c in self.children ]
        return res + "".join(children)
    
    def to_dict(self):
        result = {
            "value": self.value,
            "actions": self.actions.tolist(),
            "children": [],
        }
        for child in self.children:
            result["children"].append(child.to_dict())
        return result


def prune(
        s: Board, 
        id = 1,
        min_node=True, 
        alpha=-100, # maximum achievable value
        beta=100, # minimum achievable value
        depth=0,
        perfect_min_max= False,
        ):
    # assumes initial node is minimising
    if s.end:
        if s.winner==0:
            return P(value=0, depth=depth)
        elif min_node: # lost
            return P(value=10-depth, depth=depth)
        else: # won
            return P(value=-10+depth, depth=depth)
    else:
        node = P(actions = np.where(s.state.flatten()==0)[0], depth = depth)
        if perfect_min_max:
            beta = 100
            alpha = -100
        for action in node.actions:
            AM = np.zeros(9)
            AM[action]=id
            sn = s.next(AM.reshape(3,3))
            if min_node: # beta is local value, alpha is upstream value
                res = prune(sn, id*-1, False, alpha, beta, depth+1, perfect_min_max=perfect_min_max)
                beta = min(res.value, beta) # type:ignore
            else: # is max_node, alpha is local value, beta is upstream value
                res = prune(sn, id*-1, True, alpha, beta, depth+1, perfect_min_max=perfect_min_max)
                alpha = max(res.value, alpha) # type:ignore
            node.append(res)
            if beta<=alpha: # prune when, case min-node: upstream value is greator than current
                break       # case max-node: upstream value is lessor than current
        if min_node:
            node.value=beta
        else:
            node.value=alpha
        return node

def infer(s: Board, id=1):
    p=prune(s, id=id)
    child_values = [c.value for c in p.children]
    # print(child_values)
    AM = np.zeros(9)
    AM[p.actions[child_values.index(min(child_values))]]=id # type:ignore
    return AM.reshape(3,3)

def value(s: Board):
    p=prune(s)
    child_values = [c.value for c in p.children]
    return min(child_values) # type:ignore

if __name__ == "__main__":
    from benchmark import benchmark
    benchmark(infer, n_runs=20)
    from play import play
    s0 = Board()
    root=infer(s0, 1)
    play(infer)