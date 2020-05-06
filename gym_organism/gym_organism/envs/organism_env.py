import sys
from contextlib import closing
from io import StringIO
from gym.envs.toy_text import discrete
from gym import utils, Env, spaces

import numpy as np


MAP = [
    #x-axis: row wise, 
    #y-axis: column wise
    "+-------------------+",
    "|x: | : : : : : : : |",
    "| : | : : : : | : | |",
    "| : : : : : : : : : |",
    "| | : | : : | : :x: |",
    "|x| : | : : : : : | |",
    "| : : : : : | : : : |",
    "| : : : | : | : : : |",
    "| : | : : | : : : | |",
    "| : : : : : : : : : |",
    "| : | : : : : : | :x|",
    "+-------------------+",
]

class OrganismEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.locs = locs = [(0,0), (4,0), (3,8), (9,9)]
        num_states = 10*10*4 #food destinations are 4
        num_rows = 10
        num_columns = 10
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 5
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                    for food_idx in range(len(locs)):
                        state = self.encode(row, col, food_idx)
                        initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col = row, col
                            reward = -1 # default reward when there is no pickup/dropoff
                            done = False
                            organism_loc = (row, col)
                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup food
                                if (organism_loc == locs[food_idx]):
                                    done = True
                                    reward = 20
                                else: # pickup food at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, food_idx)
                            P[state][action].append(
                                (1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)
            
    def encode(self, organism_row, organism_col, food_idx):
        # 10, 10, 4
        i = organism_row
        i *= 10
        i += organism_col
        i *= 4
        i += food_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        organism_row, organism_col, food_idx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        out[1 + organism_row][2 * organism_col + 1] = utils.colorize(
                out[1 + organism_row][2 * organism_col + 1], 'yellow', highlight=True)
        di, dj = self.locs[food_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup Food"][self.lastaction]))
        else: outfile.write("\n")
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
