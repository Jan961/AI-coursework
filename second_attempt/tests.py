from Grid import Grid
import numpy as np

grid = Grid(4)


grid.cleave_rectangle(0, 2, vertical=True)
grid.cleave_rectangle(0,1, vertical=False)
# grid.merge_rectangles(1,0 , vertical=False)
# grid.cleave_rectangle(1, 1, vertical=True)
# grid.merge_rectangles(1,2 , vertical=True)
# grid.cleave_rectangle(1,3, vertical=False)
# grid.merge_rectangles(1,2, vertical=False)
# grid.cleave_rectangle(0,1, vertical=False)
# grid.cleave_rectangle(1,1, vertical=False)
# grid.cleave_rectangle(0,1,vertical=True)



print(f"rectangle_list: {grid.rectangle_list}")
print(f"coord list: {grid.coords_list}")
print(grid.get_state())
print(f" to merge: {grid.get_lists_to_merge()}")

print(grid.grid)
grid.show_grid()