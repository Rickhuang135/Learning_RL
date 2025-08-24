from random import random
from random import randint
max_moves = 20
gamma = 0.8
learning_rate = 0.1

class Env:
    def __init__(self, width: int, height: int, destination: tuple, walls: list[tuple]):
        self.destination = destination
        self.width = width
        self.height = height
        self.walls = walls
        self.move_count = 0
    def move_to(self, point: tuple) -> tuple:
        game_end = False
        reward = -1
        x,y = point
        if point == self.destination:
            game_end = True
            reward = 0
        elif x < 0 or x >= self.width:
            game_end = True
            reward = -1000
        elif y < 0 or y >= self.height:
            game_end = True
            reward = -1000
        else:
            for w in self.walls:
                if w==point:
                    game_end = True
                    reward = -1000
                    break
        if game_end or self.move_count== max_moves-1:
            game_end = True
            self.move_count =0
        else:
            self.move_count+=1
        return (game_end, reward)
    
    def render(self, cur_loc: tuple) -> None:
        output = ""
        cur_line=[]
        front_pad = "   "
        print(f" {front_pad}", end="")
        print(" ".join([str(x) for x in range(self.width)]))
        print()
        for y in range(self.height):
            for x in range(self.width):
                if (x,y) in self.walls:
                    cur_line.append("w")
                elif (x,y) == self.destination:
                    cur_line.append("d")
                elif (x,y) == cur_loc:
                    cur_line.append("x")
                else: 
                    cur_line.append("-")
            output=f"{output}{y}{front_pad}{' '.join(cur_line)}\n"
            cur_line=[]
        print(output)

class Agent:
    def __init__(self, start_tile:tuple, env:Env):
        self.loc = start_tile
        self.env = env
        self.Q = Action_value()
        self.visited_tiles = [start_tile]
        self.actions = []
        self.rewards = []

    # clockwise starting from north
    def move(self, action):
        x_0, y_0 = self.loc
        x_1 = y_1 = ...
        match action:
            case 0:
                x_1 = x_0
                y_1 = y_0-1
            case 1:
                x_1 = x_0+1
                y_1 = y_0
            case 2:
                x_1 = x_0
                y_1 = y_0+1
            case _:
                x_1 = x_0-1
                y_1 = y_0
        self.loc = (x_1,y_1)
        game_end, reward = self.env.move_to(self.loc)
        self.rewards.append(reward)
        self.actions.append(action)
        self.visited_tiles.append(self.loc)
        return game_end

    def test(self) -> None:
        inputt = ""
        while inputt != "q":
            action = int(input())
            game_end = self.move(action)
            self.env.render(self.loc)
            if game_end:
                inputt = "q"
                print(self.visited_tiles)
                print(self.rewards)

    def monte_carlo(self, epsilon: float) -> None:
        game_end = False
        while not game_end:
            cur_loc = self.loc
            expected_returns = [self.Q(cur_loc,x) for x in range(4)] #ensures Q remembers the current location
            if random() < epsilon: #random choose move
                a = randint(0,3)
                game_end = self.move(a)
            else:
                game_end = self.move(expected_returns.index(max(expected_returns)))
        print(f"visted_tiles:{self.visited_tiles}")
        print(f"reward:{self.rewards}")
        print(f"actions:{self.actions}")
        returns = [self.rewards.pop()]
        for r in self.rewards:
            returns.append(returns[-1]*gamma+r)
        returns = returns[::-1]
        print(f"returns:{returns}")
        print()
        for state, action, ret in zip(self.visited_tiles, self.actions, returns):
            old_val = self.Q(state, action)
            new_val = old_val+ learning_rate*(ret -old_val)
            self.Q.update(state, action, new_val)
        self.reset()

    def Q_learning(self, epsilon: float) -> None:
        game_end = False
        Q_t = [self.Q(self.loc,x) for x in range(4)]
        while not game_end:
            s_t = self.loc
            a_t = ...
            if random() < epsilon: #random choose move
                a_t = randint(0,3)
            else:
                a_t = Q_t.index(max(Q_t))
            game_end = self.move(a_t)
            r = self.rewards[-1]
            if game_end:
                new_Q_sa_t = Q_t[a_t]+ learning_rate* (r - Q_t[a_t])
                self.Q.update(s_t, a_t, new_Q_sa_t)
            else:
                Q_t1 = [self.Q(self.loc,x) for x in range(4)]
                new_Q_sa_t = Q_t[a_t]+ learning_rate* (r + gamma*max(Q_t1) - Q_t[a_t])
                self.Q.update(s_t, a_t, new_Q_sa_t)
                Q_t = Q_t1
        print(f"visted_tiles:{self.visited_tiles}")
        print(f"reward:{self.rewards}")
        print(f"actions:{self.actions}")
        self.reset()
            

    def reset(self):
        self.loc = self.visited_tiles[0]
        self.visited_tiles=[self.loc]
        self.actions.clear()
        self.rewards.clear()

class State:
    def __init__(self, uid: tuple):
        self.uid = uid
        self.action_returns = [0.0 for _ in range(4)]

    def get(self, action: int)->float:
        return self.action_returns[action]

    def update(self, action:int, value ) -> None:
        self.action_returns[action] = value

    def __eq__(self, other):
        if isinstance(other, State):
            return self.uid == other.uid
        if isinstance(other, tuple):
            return self.uid == other
        return False
    
    def __str__(self):
        return f"{self.uid}: {self.action_returns}"

class Action_value:
    def __init__(self):
        self.states: list[State] = []
    def __call__(self, point: tuple, action:int):
        for existing_state in self.states:
            if existing_state == point:
                state = existing_state
                return state.get(action)
        state = State(point)
        self.states.append(state)
        return state.get(action)
    
    def update(self, point:tuple, action, value):
        for existing_state in self.states:
            if existing_state == point:
                existing_state.update(action, value)
                return
        raise Exception(f"point {point} not found in available states")

def evaluate():
    # easy
    # map = Env(3,3,(2,2),[(1,1)])
    # start = (0,0)

    # normal
    # map = Env(5,3,(4,2),[(2,1),(2,2)])
    # start = (0,2)

    # difficult
    map = Env(8,5,(7,4), [(2,0),(2,1),(2,2),(2,3),(5,1),(5,2),(5,3),(5,4)])
    start = (0,0)

    you = Agent(start, map)
    map.render(you.loc)
    epsilon = 0.9
    total_runs = 5000
    step_size = epsilon*10/total_runs
    for x in range(total_runs):
        # you.monte_carlo(epsilon)
        you.Q_learning(epsilon)
        if x % 10 ==0:
            epsilon -= step_size
            print(epsilon)
    map.render(start)
    for state in you.Q.states:
        print(state)
    # you.monte_carlo(epsilon)
    you.Q_learning(epsilon)

evaluate()