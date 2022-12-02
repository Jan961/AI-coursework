from Grid import Grid
import numpy as np

grid = Grid(4)


grid.split_rectangle(0, 1)




grid.split_rectangle(1,3, vertical=False)

print(grid.available_dimensions)
print(grid.no_rectangles)
print(f"rectangle_list: {grid.rectangle_list}")
print(f"coord list: {grid.coords_list}")
print(grid.get_Mond_score())
print(grid.get_state())

grid.show_grid()