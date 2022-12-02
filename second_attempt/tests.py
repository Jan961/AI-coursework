from Grid import Grid
import numpy as np

grid = Grid(4)


grid.cleave_rectangle(0, 2, vertical=True)
grid.cleave_rectangle(0,2, vertical=False)





print(grid.available_dimensions)
print(grid.no_rectangles)
print(f"rectangle_list: {grid.rectangle_list}")
print(f"coord list: {grid.coords_list}")
print(grid.get_state())

print(grid.grid)
grid.show_grid()