from Grid import Grid
import numpy as np
from actions import split, merge

grid = Grid(50)

play = input('type anything to start')
grid.show_grid()


while True:
    choice = input('split or merge? \n type "s" or "m"')
    if choice =='s':
        split(grid, random=True)
    elif choice =='m':
        merge(grid, random=True)
    else:
        break

    grid.show_grid()
    print(f"rectangle_list: {grid.rectangle_list}")
    print(f"coord list: {grid.coords_list}")
    print(grid.get_grid_state())
    print(grid.grid)





# print(f"rectangle_list: {grid.rectangle_list}")
# print(f"coord list: {grid.coords_list}")
# print(grid.get_grid_state())
# print(grid.grid)
