import numpy as np

from BestFirstWithAnnealing import BestFirstWithAnnealing
from Grid import Grid

class RandomSearch():
    def __init__(self, epochs,  **kwargs):
        self.exponential_annealing_schedule = kwargs['exponential_annealing_schedule']
        self.annealing_params = kwargs['annealing_params']
        self.grid_size = kwargs['grid_size']
        self.sample_size = kwargs['sample_size']
        self.epochs = epochs

    def search(self):
        scores = []
        for i in range(self.epochs):
            one_search = BestFirstWithAnnealing(self.grid_size, self.sample_size, self.annealing_params,
                                                self.exponential_annealing_schedule)
            one_search_result = one_search.run()

            if one_search_result == -1:
                print("stuck")
            else:
                print("found sol")
                scores.append(one_search_result)
        print( scores)
        print(f"best Mondrian sc: {sorted(scores)[0]}")
        print(f"completed grids:{len(scores)}")
        print(f"mean : {np.mean(np.array(scores))}")
        print(f"std : {np.std(np.array(scores))}")


