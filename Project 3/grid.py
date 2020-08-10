import numpy as np


class Grid:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def move(self, dire, index):

        self.i = index[0]
        self.j = index[1]

        if dire in self.actions[(self.i, self.j)]:
            if dire == 'U':
                self.i -= 1
            elif dire == 'L':
                self.j -= 1
            elif dire == 'D':
                self.i += 1
            elif dire == 'R':
                self.j += 1

        state1 = (self.i, self.j)
        if self.rewards.get((self.i, self.j)) == None:
            return 0,state1
        else:
            return self.rewards.get((self.i, self.j)),state1

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

def every_action(gridworld):
    all_actions = {}
    i, j = np.where(gridworld == 'X')
    for index in range(0, len(i)):
        a = []
        if j[index] + 1 < len(gridworld):
            a.append("R")
        if j[index] - 1 >= 0:
            a.append("L")
        if i[index] - 1 >= 0:
            a.append("U")
        if i[index] + 1 < len(gridworld):
            a.append("D")
        all_actions[(i[index], j[index])] = tuple(a)

    return all_actions


def values_output(value, make_grid):
    for i in range(make_grid[0]):
        print("_____________________________________________________________")
        for j in range(make_grid[1]):
            v = value.get((i, j))
            print(" %.2f |" % v, end="")
        print("")


def every_rewards(rewards):
    all_rewards = {}
    il, jl = np.where(rewards != 'X')
    for index in range(0, len(il)):
        all_rewards[(il[index], jl[index])] = int(rewards[il[index]][jl[index]])
    return all_rewards


def policy_output(policy, make_grid):
    for i in range(make_grid[0]):
        print("___________________________")
        for j in range(make_grid[1]):
            a = policy.get((i,j), ' ')
            print(" %s |" % a, end="")
        print("")
    print("")

def matrix(all_actions, all_rewards):
    grid = Grid(0, 0)
    grid.set(all_rewards, all_actions)
    return grid


