from Grid import Grid
import numpy as np
from actions import split, merge

grid = Grid(15)


split(grid, random=True)






grid.show_grid()









print(f"rectangle_list: {grid.rectangle_list}")
print(f"coord list: {grid.coords_list}")
print(grid.get_grid_state())


print(grid.grid)
# grid.show_grid()