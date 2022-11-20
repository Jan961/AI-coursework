import numpy as np
from BestFirstWithAnnealing import BestFirstWithAnnealing
from Grid import Grid


def main():
    bestfirst = BestFirstWithAnnealing(grid_size=6, sample_size=20, annealing_params=[0.5,-5] )
    bestfirst.run()
    print("rec id: ", bestfirst.grid.rectangle_id)


















if __name__ == "__main__":
    main()