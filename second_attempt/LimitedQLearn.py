import copy

from Grid import Grid
import numpy as np
from actions import split, merge
from collections import deque


# Limited Q Learning
class LimitedQLearn:

    def __init__(self, grid_size =8, max_steps = 10 ,  epochs= 50, gamma = 0.1,
                 noise = 0.08, noise_decay = 0.1, reward_scaling = (1,1) ):

        self.max_steps = max_steps  # max number of steps before terminating the epoch
        # q-table: indexed by State: num rectangles, Mondrian score, and actions
        self.qtable = np.random.rand(grid_size**2, grid_size**2, 5)
        self.grid_size= grid_size
        self.grid = Grid(grid_size)
        self.epochs = epochs
        self.gamma = gamma
        self.noise = noise
        self.noise_decay = noise_decay
        self.reward_scaling = reward_scaling
        self.two_actions = deque([(True,True),(False,True)]) # e.g. [(True,True)(False,True)] a queue of two consecutive action results
        # 1st number in the tuple indicates action: merge or split second if it were successful
        # an unsuccessful merge followed by an unsuccessful split means stuck
        self.best = []   #best results found




    #indices for the q table
    def indices_from_state(self, state):
        return state.no_rectangles, state.Mond_score

    # back to starting position - num rect, Mondrian score and reward: -10
    def reset(self):
        self.grid = Grid(self.grid_size)
        return self.grid.get_grid_state(), -10

    def random_action(self):
        return np.random.choice([0,1,2,3,4])

    def check_is_stuck(self):
        if self.two_actions[0][0] == self.two_actions[1][0]:
            return False
        if self.two_actions[0][1] or self.two_actions[1][1]:
            return False

        return True

    def calc_reward(self, state):
        sc =  state.Mond_score
        a = self.reward_scaling[0]
        b =self.reward_scaling[1]
        return np.exp(-a * sc + b)



    def step(self, action):

        if action == 0:
            self.two_actions.popleft()
            outcome = split(self.grid)
            self.two_actions.append((True, outcome))

        if action == 1:
            self.two_actions.popleft()
            outcome = split(self.grid, random=True)
            self.two_actions.append((True, outcome))

        if action == 2:
            self.two_actions.popleft()
            outcome = merge(self.grid)
            self.two_actions.append((False, outcome))

        if action == 3:
            self.two_actions.popleft()
            outcome = merge(self.grid, random=True)
            self.two_actions.append((False, outcome))

        if action == 4:
            self.two_actions.popleft()
            outcome = True
            self.two_actions.append((False, outcome))

        self.grid.show_grid()
        next_state = self.grid.get_grid_state()
        reward = self.calc_reward(next_state)

        return next_state, reward



    # training loop
    def train(self):
        best_score = self.grid_size**2
        best_grid = None
        for i in range(self.epochs):
            state, reward = self.reset()
            steps = 0

            while steps < self.max_steps and not self.check_is_stuck():
                steps += 1

                # exploration
                if np.random.uniform() < self.noise:
                    action = self.random_action()
                else:
                    no_r, Mond = self.indices_from_state(state)
                    action = np.argmax(self.qtable[no_r, Mond, :])
                print(f"action {action}")
                #  take action
                next_state, reward = self.step(action)

                # update qtable value with Bellman equation
                no_r, Mond = self.indices_from_state(state)
                no_r2, Mond2 = self.indices_from_state(next_state)
                self.qtable[no_r, Mond, action] = reward + self.gamma * \
                                                  np.max(self.qtable[no_r2, Mond2,:])

                # update state
                state = next_state
                if best_score > state.Mond_score:
                    best_score = state.Mond_score
                    best_grid = copy.deepcopy(self.grid)
            # limit exploration
            self.noise -= self.noise_decay * self.noise

            print(f"Done in {steps}steps,  epoch: {i}, Best mond score: {best_score}" )

        print(f"all epochs done, best score: {best_score} ")
        # best_grid.show_grid()
