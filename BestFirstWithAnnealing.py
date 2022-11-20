import matplotlib.pyplot as plt
from Rectangle import Rectangle
from Grid import Grid
import numpy as np

class BestFirstWithAnnealing:
    def __init__(self, grid_size, sample_size,
                 annealing_params, exponential_annealing_schedule = False):
        self.grid = Grid(grid_size)
        self.sample_size = sample_size
        self.exponential_annealing_schedule = exponential_annealing_schedule

        #annealing params is a list
        # for linear annealing schedule the first is the intercept, second gradient
        # for exponential 1st is "a" and the 2nd "b" in the formula a * exp(-k + b)
        self.param_1 = annealing_params[0]
        self.param_2 = annealing_params[1]

        #param 1 has to be >0 for both exp and linear schedules
        assert self.param_1 > 0

        #for linear schedules the grad has to be <0
        if not self.exponential_annealing_schedule:
            assert self.param_2 < 0


    def one_rectangle_addition(self):

        if self.grid.rectangle_id == 1:
            self.grid.add_rectangle(self.grid.create_new_rectangle())
        else:
            sample = self._create_sample()
            print(f"potential scores: {sample[:,-1]}" )
            temperature = self._get_temperature()
            added_temp = self._add_temperature(sample, temperature)
            print(f"potential scores + temp: {added_temp[:, -1]}")
            best_rectangle = self._choose_best_rectangle(added_temp)

            self.grid.add_rectangle(best_rectangle)

    def run(self):
        mond_scores = [0]
        temperatures = []
        while self.grid.candidate_places.size != 0:
            self.one_rectangle_addition()
            if self.grid.rectangle_id > 1:
                mond_score = self.grid.get_mondrian_score()
                print(f"Mondrian score: {mond_score}" )
                mond_scores.append(mond_score)

            temperatures.append(self._get_temperature())
            self.grid.show_grid()

        print(f"Final Mondrian score: {self.grid.get_mondrian_score()}")
        plt.figure()
        plt.plot(np.arange(self.grid.rectangle_id), mond_scores, 'b', label="Mondrian score")
        plt.plot(np.arange(self.grid.rectangle_id-1), temperatures, 'r', label="Temperature")
        plt.show()

    def _create_sample(self):
        rectangles = set(self.grid.create_new_rectangle() for i in range(self.sample_size))
        # each rectangle is a row of 1st point x,1st point y, width, height, area
        rectangles_arr  = np.stack((r.to_arr() for r in rectangles), axis=0 )
        scores = np.array(list(map(self.grid.get_potential_mond_score, rectangles_arr[:,4])))[:, None]
        return np.concatenate((rectangles_arr,scores), axis=1)


    def _add_temperature(self, rectangles, variance):
        rectangles[:,5] += variance * np.random.randn(rectangles.shape[0])
        return rectangles

    def _choose_best_rectangle(self, rectangles_arr):
        index_of_best = np.argmin(rectangles_arr[:,5])
        best_rectangle =np.squeeze(rectangles_arr[ rectangles_arr[:,5] == rectangles_arr[index_of_best, 5]])
        if best_rectangle.ndim>1:
            best_rectangle = best_rectangle[np.random.randint(0, best_rectangle.shape[0])]

        print(f"best recangles socre: { best_rectangle[5]} ", )
        return Rectangle(*best_rectangle[:4].tolist())

    def _get_temperature(self):
        if self.exponential_annealing_schedule:
            temperature = self._get_exp_schedule_variance()
        else:
            temperature = self._get_linear_schedule_variance()

        return temperature


    def _get_linear_schedule_variance(self):
        variance = self.param_1 + self.param_2 * (self.grid.rectangle_id - 1)
        return variance if variance > 0 else 0

    def _get_exp_schedule_variance(self):
        return self.param_1 * np.e ** ((self.grid.rectangle_id - 1) + self.param_2)








