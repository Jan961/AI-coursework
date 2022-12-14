#general structure of the q-learning code adapted from:
#https://towardsdatascience.com/math-of-q-learning-python-code-5dcbdc49b6f6



import copy
import matplotlib.pyplot as plt
from Grid import Grid
import numpy as np
from actions import split, merge
from collections import deque
import time


# Limited Q Learning
class LimitedQLearn:

    def __init__(self, grid_size =8, max_steps = 10 ,  epochs= 50, gamma = 0.1,
                 noise = 0.08, noise_decay = 0.1, reward_scaling = (1,1), learning_rate=0.6, allow_do_nothing = True ):

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
        self.learning_rate= learning_rate
        self.two_actions = deque([(True,True),(False,True)]) # e.g. [(True,True)(False,True)] a queue of two consecutive action results
        # 1st number in the tuple indicates action: merge or split; second if it were successful
        # an unsuccessful merge followed by an unsuccessful split or vice versa means stuck
        # this check was added to enable allowing invalid states
        self.allow_do_nothing = allow_do_nothing




    #indices for the q table
    def indices_from_state(self, state):
        return state.no_rectangles, state.Mond_score

    # back to starting position - num rect, Mondrian score and reward: -10
    def reset(self):
        self.grid = Grid(self.grid_size)
        self.two_actions = deque([(True,True),(False,True)])
        return self.grid.get_grid_state(), -10

    def check_is_stuck(self):
        if self.two_actions[0][0] == self.two_actions[1][0]:
            return False
        if self.two_actions[0][1] or self.two_actions[1][1]:
            return False

        return True

    def calc_reward(self, state):
        if state.no_rectangles == 1:
            return -100
        sc =  state.Mond_score
        a = self.reward_scaling[0]
        b =self.reward_scaling[1]
        # return a*np.log(-sc + b)
        return np.exp(- a * sc + b)



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

        next_state = self.grid.get_grid_state()
        reward = self.calc_reward(next_state)

        return next_state, reward



    # training loop
    def train(self):
        start= time.perf_counter()
        initial_mondrian = self.grid_size**1.2
        terminal_mond_scores = [initial_mondrian] #initial value inside- not too large to make the graph look good
        best_grid = None
        best_scores = [initial_mondrian] #initial value inside - not too large to make the graph look good
        for i in range(self.epochs):
            state, reward = self.reset()
            steps = 0
            epoch_best = best_scores[-1]

            # print(f"is stuck: {self.check_is_stuck()}")
            while steps < self.max_steps and not self.check_is_stuck():
                steps += 1

                no_r, Mond = self.indices_from_state(state)

                if self.allow_do_nothing and no_r > 1: #if we allow action "do nothing" also no "do nothing" in initial state
                    action_indices = [0,1,2,3,4]
                else:
                    action_indices = [0, 1, 2, 3]


                if np.random.uniform() < self.noise: # exploration
                    action = np.random.choice(action_indices)
                else:
                    action = np.argmax(self.qtable[no_r, Mond, action_indices])

                # print(f"action {action}")

                next_state, reward = self.step(action)

                # update qtable value with Bellman equation
                no_r, Mond = self.indices_from_state(state)
                no_r2, Mond2 = self.indices_from_state(next_state)
                curr_q = self.qtable[no_r, Mond, action]
                self.qtable[no_r, Mond, action] = curr_q + self.learning_rate\
                                                  *( self.calc_reward(next_state)
                                                     + self.gamma * np.max(self.qtable[no_r2, Mond2, action_indices])
                                                     - curr_q)

                # update state
                state = next_state
                if epoch_best > state.Mond_score:
                    epoch_best = state.Mond_score
                    best_grid = copy.deepcopy(self.grid)



            # limit exploration
            best_scores.append(epoch_best)
            self.noise -= self.noise_decay * self.noise
            terminal_mond_scores.append(state.Mond_score)
            print(f"Done in {steps}steps,  epoch: {i}, Terminal mond score: {terminal_mond_scores[-1]}, Best score: {best_scores[-1]}" )


        print(f"Finished: {best_scores[-1]} ")
        print(f"Time taken: {time.perf_counter() - start}")
        best_grid.show_grid()
        print(f"best grid score{best_grid.get_Mond_score()}")
        print(f"grid: ")
        print(best_grid.grid)
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, self.epochs+1), best_scores, label="Best score", c='r')
        running_mean = np.convolve(terminal_mond_scores, np.ones(30)/30, mode='same')
        ax.plot(np.arange(0, self.epochs+1),running_mean, label="Score at the end of an epoch", c='b')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Mondrian score")
        ax.set_ylim(0,initial_mondrian*2)
        plt.legend()
        plt.show()


