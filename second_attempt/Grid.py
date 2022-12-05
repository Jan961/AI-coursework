import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from GridState import GridState
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
        if self.no_rectangles == 1:
            return False

        ordered = np.sort(self.rectangle_list[:,:2], axis=1)
        # hard-coded 100 but could easily be adapted to work for any grid size if I started this exercise earlier
        added = ordered[:,0]*100 + ordered[:,1]
        return np.unique(added).size == added.size


    def get_grid_state(self):
        return GridState(self.no_rectangles, self.get_Mond_score(), self.is_valid)

    def get_Mond_score(self):
        if self.no_rectangles == 1:
            return self.size**2 - 1
        else:
            return np.max(self.rectangle_list[:,2]) - np.min(self.rectangle_list[:,2])

    #args are: rectangle_id, first_side, vertical if split == True
    # frist_rcetangle_id, second_recangle_id, vertical if split == False
    def get_potential_grid_state(self, *args, split=True):

        rect_list = copy.deepcopy(self.rectangle_list)

        if split:
            h1, w1, h2, w2 = self._get_new_dims_split(args[0], args[1], args[2])

            rect_list[args[0]][0], rect_list[args[0]][1],rect_list[args[0]][2] = h1, w1, h1*w1
            rect_list = np.insert(rect_list, 0, [h2, w2, h2*w2], axis=0)

        else:
            new_h, new_w = self._get_new_dims_merge(args[0], args[1], args[2])

            rect_list[args[0]][0], rect_list[args[0]][1], rect_list[args[0]][2] = new_h, new_w, new_h*new_w
            rect_list = np.delete(rect_list, args[1], 0)

        Mondrian = np.max(rect_list[:,2]) - np.min(rect_list[:,2])
        valid = self._check_list_validity(rect_list)
        no_rectangles= rect_list.shape[0]

        return GridState(no_rectangles, Mondrian, valid)

    # I started too late to have time to think how to optimise the below function

    # here and in all other functions below horizontal merge means, perhaps counterintuitively,
    # that the dividing line that disappears when the rectangles are merged is horizontal
    #and likewise for vertical
    def get_lists_to_merge(self):
        vertical = []
        horizontal = []
        for i in range(self.coords_list.shape[0]):
            c11 = self.coords_list[i,2]
            c12 = self.coords_list[i,3]
            r11 = self.coords_list[i,0]
            r12 = self.coords_list[i,1]
            for j in range(self.coords_list.shape[0]):
                if j == i:
                    continue
                c21 = self.coords_list[j, 2]
                c22 = self.coords_list[j, 3]
                r21 = self.coords_list[j, 0]
                r22 = self.coords_list[j, 1]

                if c11 == c21 and c12 == c22 and r11 == r22 + 1:
                    horizontal.append([i,j])
                elif r11 == r21 and r12 == r22 and c12 +1 == c21:
                    vertical.append([i,j])

        return vertical, horizontal




    # not checking if the paramentes are valid - e.g. new_side > old side
    # rectangle id indexed from 0
    def cleave_rectangle(self, rectangle_id, new_side1, vertical):

        self.grid = np.where(self.grid > rectangle_id, self.grid +1, self.grid)
        old_s = self.rectangle_list[rectangle_id][:2] #sides (dimensions) of the old rectangle

        self.no_rectangles += 1

        h1, w1, h2, w2 = self._get_new_dims_split(rectangle_id, new_side1, vertical)

        if vertical:
            self.coords_list[rectangle_id][3] = self.coords_list[rectangle_id][2] + w1 - 1
            coords = self.coords_list[rectangle_id] # coords of the new rectangle on the left
            self.grid[coords[0]:coords[1]+1,coords[3] + 1:coords[3] + 1 + w2] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id +1, [coords[0], coords[1], coords[3] + 1, coords[3] + w2 ], axis=0)


        else:
            self.coords_list[rectangle_id][1] = self.coords_list[rectangle_id][0] + h1 - 1
            coords = self.coords_list[rectangle_id]  # coords of the new rectangle at the top
            self.grid[coords[1]+1: coords[1] + 1 + h2, coords[2] :coords[3] + 1] = rectangle_id + 1
            self.coords_list = np.insert(self.coords_list,
                      rectangle_id + 1, [coords[1]+1, coords[1] + h2, coords[2], coords[3]], axis=0)


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
            row_coords = np.sort(np.concatenate((self.coords_list[r_id_1][:2],
                                                    self.coords_list[r_id_2][:2])))[[0, -1]]


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

        my_cmap = matplotlib.cm.get_cmap('gist_rainbow')
        ax.imshow(self.grid, vmin=0, vmax=self.no_rectangles -1, cmap=my_cmap)
        plt.show()



    def _initialise_coord_list(self):
        x = 0
        y = self.size -1
        return np.array([[x, y, x, y]])

    def _check_list_validity(self, rectangle_list):
        # print(f"rectangle list: {rectangle_list}, shape: {rectangle_list.shape[0]}" )
        if rectangle_list.shape[0] == 1:
            return False
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
            new_h = self.rectangle_list[r_id_1][0]
            new_w = self.rectangle_list[r_id_1][1] +  self.rectangle_list[r_id_2][1]
        else:
            new_h = self.rectangle_list[r_id_1][0] +  self.rectangle_list[r_id_2][0]
            new_w = self.rectangle_list[r_id_1][1]

        return new_h, new_w








