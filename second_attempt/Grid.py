import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size)).astype(int)
        self.corners_list = self._initialise_corners_list()
        self.corners_grid = self._initialise_corners_grid()
        self.available_dimensions = self._initialise_dimensions()
        self.rectangle_list = np.array([size, size, size*size])[None,:]
        self.no_rectangles = 1


    def _initialise_corners_grid(self):
        c = np.full((self.size, self.size),np.NaN)
        for i in range(4):
            c[self.corners_list[0][i]] = 0
        return c

    def _initialise_dimensions(self):
        d =np.full((self.size, self.size), True)
        d[self.size -1, self.size-1] = False
        return d
    def _initialise_corners_list(self):
        x = np.array([0])
        y = np.array([self.size -1])
        return [[(x, x), (x, y), (y,y), (y,x)]]





    def split_rectangle(self, rectangle_id, new_side_left, vertical= True):
        self.grid = np.where(self.grid > rectangle_id, self.grid +1, self.grid)

        
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


