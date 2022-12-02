import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from State import State

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size)).astype(int)
        self.coords_list = self._initialise_coord_list() # each entry: top row coord, bottom row, left column, right col
        self.available_dimensions = self._initialise_dimensions()
        self.rectangle_list = np.array([size, size, size*size])[None,:] # each entry: height, width, area
        self.no_rectangles = 1 # = rectangle id +1
        self.is_valid = True

    def get_state(self):
        return State(self.no_rectangles, self.get_Mond_score(), self.is_valid)

    def get_Mond_score(self):
        return np.max(self.rectangle_list[:,2]) - np.min(self.rectangle_list[:,2])


    # no checking if the paramentes are right - e.g. new_side > old side
    # rectangle id indexed from 0
    def split_rectangle(self, rectangle_id, new_side1, vertical= True):

        self.grid = np.where(self.grid > rectangle_id, self.grid +1, self.grid)
        old_s = self.rectangle_list[rectangle_id][:2] #sides (dimensions) of the old rectangle


        if self.no_rectangles > 1:
            self.available_dimensions[[old_s[0]-1, old_s[1]-1], [old_s[1]-1, old_s[0]-1]] = True

        self.no_rectangles += 1

        if not self._check_validity_split(old_s, new_side1, vertical):
            self.is_valid = False


        if vertical:
            new_side2 = self.rectangle_list[rectangle_id][1] - new_side1
            self.coords_list[rectangle_id][3] = self.coords_list[rectangle_id][2] + new_side1 - 1
            coords = self.coords_list[rectangle_id] # coords of the new rectangle on the left
            self.grid[coords[0]:coords[1]+1,coords[3] + 1:coords[3] + 1 + new_side2] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id +1, [coords[0], coords[1], coords[3] + 1, coords[3] + new_side2 ], axis=0)

            self.available_dimensions[[old_s[0]-1, new_side1-1], [new_side1-1, old_s[0]-1]] = False
            self.available_dimensions[[old_s[0]-1, new_side2-1],[new_side2-1, old_s[0]-1]] = False

            self.rectangle_list[rectangle_id][1] = new_side1
            self.rectangle_list[rectangle_id][2] = new_side1 * old_s[0]

            self.rectangle_list = np.insert(self.rectangle_list,
                                            rectangle_id+1, [old_s[0], new_side2, new_side2*old_s[0]], axis=0)


        else:
            new_side2 = self.rectangle_list[rectangle_id][0] - new_side1
            self.coords_list[rectangle_id][1] = self.coords_list[rectangle_id][0] + new_side1 - 1
            coords = self.coords_list[rectangle_id]  # coords of the new rectangle at the top
            self.grid[coords[1]+1: coords[1] + 1 + new_side2, coords[2] :coords[3] + 1] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id + 1, [coords[1]+1, coords[1] + new_side2, coords[2], coords[3]], axis=0)

            self.available_dimensions[[old_s[1]-1, new_side1-1], [new_side1-1, old_s[1]-1]] = False
            self.available_dimensions[[old_s[1]-1, new_side2-1], [new_side2-1, old_s[1]-1]] = False

            self.rectangle_list[rectangle_id][0] = new_side1
            self.rectangle_list[rectangle_id][2] = new_side1 * old_s[1]

            self.rectangle_list = np.insert(self.rectangle_list, rectangle_id + 1,
                                            [new_side2, old_s[1], new_side2 * old_s[1]], axis=0)









        
    def show_grid(self):

        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        ax.set_xlim(-0.5, self.size -0.5)
        ax.set_ylim(-0.5, self.size -0.5)

        labels = [str(i) for i in range(1, self.size+1)]
        ax.set_xticks([i for i in range(self.size)])
        ax.set_yticks([i for i in range(self.size)])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        my_cmap = matplotlib.cm.get_cmap('rainbow')
        ax.imshow(self.grid, vmin=0, vmax=self.no_rectangles -1, cmap=my_cmap)
        plt.show()


    def _initialise_dimensions(self):
        d =np.full((self.size, self.size), True)
        d[self.size -1, self.size-1] = False
        return d

    def _initialise_coord_list(self):
        x = 0
        y = self.size -1
        return np.array([[x, y, x, y]])

    def _check_validity_split(self, old_dims, new_side1, vertical):
        # - 1 because the available dims grid is indexed from 0
        if vertical:
            r1 = [old_dims[0]-1, new_side1-1]
            r2 = [old_dims[0]-1, old_dims[1] - new_side1-1]

        else:
            r1 = [new_side1-1, old_dims[1]-1]
            r2 = [old_dims[1] - new_side1-1, old_dims[1]-1]

        bools = self.available_dimensions[[r1[0],r1[1],r2[0],r2[1]],[r1[1],r1[0],r2[1],r2[0]]]
        return not np.any(np.invert(bools))










