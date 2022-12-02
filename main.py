import numpy as np
from BestFirstWithAnnealing import BestFirstWithAnnealing
from Grid import Grid
from RandomSearch import RandomSearch
from time import perf_counter


def main():
    search = RandomSearch(epochs=2000, grid_size=16, sample_size=50, annealing_params=[1, +100],
                          exponential_annealing_schedule = True  )
    start = perf_counter()
    search.search()
    end = perf_counter()

    print(f"time: {start - end}")







if __name__ == "__main__":
    main()