import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from State import State
import copy

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size)).astype(int)
        self.coords_list = self._initialise_coord_list() # each entry: top row coord, bottom row, left column, right col
        self.rectangle_list = np.array([size, size, size*size])[None,:] # each entry: height, width, area
        self.no_rectangles = 1 # equals rectangle id +1
        self.is_valid = False

    def check_is_valid(self):
        ordered = np.sort(self.rectangle_list[:,:2], axis=1)
        # hard-coded 100 but could easily be adapted to work for any grid size if I started this exercise earlier
        added = ordered[:,0]*100 + ordered[:,1]
        return np.unique(added).size == added.size


    def get_state(self):
        return State(self.no_rectangles, self.get_Mond_score(), self.is_valid)

    def get_Mond_score(self):
        return np.max(self.rectangle_list[:,2]) - np.min(self.rectangle_list[:,2])

    def get_potential_state(self, split=True, *args):
        pass


    def get_potential_validity_split(self, rectangle_id, new_side1, vertical):
        rect_list = copy.deepcopy(self.rectangle_list[:,:2])

        h1, w1, h2, w2 = self._get_new_dims_split(rectangle_id, new_side1, vertical)

        rect_list[rectangle_id][0], rect_list[rectangle_id][1] = h1, w1
        rect_list = np.insert(rect_list, 0,[h2, w2], axis=0 )

        return self._check_list_validity(rect_list)



    def get_potential_validity_merge(self, r_id_1, r_id_2, vertical):
        rect_list = copy.deepcopy(self.rectangle_list[:, :2])

        new_h, new_w = self._get_new_dims_merge(r_id_1, r_id_2, vertical)

        rect_list[r_id_1][0], rect_list[r_id_1][1] = new_h, new_w
        rect_list= np.delete(rect_list, r_id_2, 0)
        return self._check_list_validity(rect_list)



    # no checking if the paramentes are valid - e.g. new_side > old side
    #assuming that the new_side 1 is always on the left or at the top
    # rectangle id indexed from 0
    def cleave_rectangle(self, rectangle_id, new_side1, vertical):

        self.grid = np.where(self.grid > rectangle_id, self.grid +1, self.grid)
        old_s = self.rectangle_list[rectangle_id][:2] #sides (dimensions) of the old rectangle

        self.no_rectangles += 1

        h1, w1, h2, w2 = self._get_new_dims_split(rectangle_id, new_side1, vertical)

        if vertical:
            new_side2 = self.rectangle_list[rectangle_id][1] - new_side1
            self.coords_list[rectangle_id][3] = self.coords_list[rectangle_id][2] + new_side1 - 1
            coords = self.coords_list[rectangle_id] # coords of the new rectangle on the left
            self.grid[coords[0]:coords[1]+1,coords[3] + 1:coords[3] + 1 + new_side2] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id +1, [coords[0], coords[1], coords[3] + 1, coords[3] + new_side2 ], axis=0)


        else:
            new_side2 = self.rectangle_list[rectangle_id][0] - new_side1
            self.coords_list[rectangle_id][1] = self.coords_list[rectangle_id][0] + new_side1 - 1
            coords = self.coords_list[rectangle_id]  # coords of the new rectangle at the top
            self.grid[coords[1]+1: coords[1] + 1 + new_side2, coords[2] :coords[3] + 1] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id + 1, [coords[1]+1, coords[1] + new_side2, coords[2], coords[3]], axis=0)


        self.rectangle_list[rectangle_id][0], self.rectangle_list[rectangle_id][1],\
        self.rectangle_list[rectangle_id][2] =  h1, w1, h1*w1

        self.rectangle_list = np.insert(self.rectangle_list, rectangle_id + 1,
                                        [h2, w2, h2 * w2], axis=0)

        self.is_valid = self.check_is_valid()


    # not checking if the parameters are valid
    def merge_rectangles(self, r_id_1, r_id_2, vertical):

        ids = sorted([r_id_1, r_id_2])
        self.grid = np.where(self.grid == ids[1], ids[0], self.grid)
        self.grid = np.where(self.grid > ids[1], self.grid -1, self.grid)

        new_h, new_w = self._get_new_dims_merge(r_id_1, r_id_2, vertical)

        if vertical:
            column_coords = np.sort(np.concatenate((self.coords_list[r_id_1][2:],
                                                    self.coords_list[r_id_2][2:])))[[0, -1]]
            row_coords = self.coords_list[r_id_1][[0, 1]]

        else:
            column_coords = self.coords_list[r_id_1][[2, 3]]
            row_coords = np.sort(np.concatenate((self.coords_list[r_id_1][:3],
                                                    self.coords_list[r_id_2][:3])))[[0, -1]]


        self.coords_list[ids[0]] = [row_coords[0], row_coords[1], column_coords[0], column_coords[1]]
        self.coords_list = np.delete(self.coords_list, ids[1], axis=0)

        self.rectangle_list[ids[0]] = [new_h, new_w, new_h*new_w]
        self.rectangle_list = np.delete(self.rectangle_list, ids[1], axis=0)
        self.no_rectangles -=1

        self.is_valid = self.check_is_valid()




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



    def _initialise_coord_list(self):
        x = 0
        y = self.size -1
        return np.array([[x, y, x, y]])

    def _check_list_validity(self, rectangle_list):
        ordered = np.sort(rectangle_list, axis=1)
        added = ordered[:, 0] * 100 + ordered[:, 1]
        return np.unique(added).size == added.size

    def _get_new_dims_split(self, rectangle_id, new_side_1, vertical):
        if vertical:
            h = self.rectangle_list[rectangle_id][0]
            new_side2 = self.rectangle_list[rectangle_id][1] - new_side_1

            return h, new_side_1, h, new_side2
        else:
            w = self.rectangle_list[rectangle_id][1]
            new_side2 = self.rectangle_list[rectangle_id][0] - new_side_1

            return new_side_1, w, new_side2, w

    def _get_new_dims_merge(self, r_id_1, r_id_2, vertical):

        if vertical:
            new_h = self.rectangle_list[r_id_1][0] +  self.rectangle_list[r_id_2][0]
            new_w = self.rectangle_list[r_id_1][0]
        else:
            new_h = self.rectangle_list[r_id_1][0]
            new_w = self.rectangle_list[r_id_1][1] +  self.rectangle_list[r_id_2][1]

        return new_h, new_w








