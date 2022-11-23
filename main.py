import numpy as np
from BestFirstWithAnnealing import BestFirstWithAnnealing
from Grid import Grid
from RandomSearch import RandomSearch


def main():
    search = RandomSearch(epochs=1000, grid_size=8, sample_size=30, annealing_params=[1, +100],
                          exponential_annealing_schedule = True  )
    search.search()








if __name__ == "__main__":
    main()