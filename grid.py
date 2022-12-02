import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Rectangle import Rectangle



class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size)).astype(int)
        self.rectangle_id = 1 # rect id shows the id of the NEXT rectangle i.e. it is 1 for an empty grid - too late to change
        self.candidate_places = self._create_list_of_candidate_plc()
        self.available_dimensions = self._initialise_dimensions()
        self.rectangle_area_list = []


    def create_new_rectangle(self, attempts=0):


        # print(f"attempt")

        while attempts < 100:

            first_point = self._select_random_first_point()
            # print("first point: ", first_point)
            column_below = self.grid[first_point[0]:, first_point[1]] #this includes the point itself (for unit height rectangles)
            # print("column below: ", column_below )
            other_rectangles_below = np.nonzero(column_below)[0]

            if other_rectangles_below.size != 0:
                free_space_below = column_below[:other_rectangles_below[0]]
            else:
                free_space_below = column_below

            # insert indices into our column of zeros to help filter available dimensions
            free_space_below = np.arange(free_space_below.size)
            available_first_dimension = self.available_dimensions.any(axis=1)[:free_space_below.size]
            # print("aval frist dim: ", available_first_dimension )
            available_space_below = free_space_below[available_first_dimension]
            # print("aval space below: ", available_space_below)

            # find a different point if there is no available space below (taking into account that some dimensions may have
            # already been used)
            if available_space_below.size == 0:
                attempts +=1
                # print("continue")
                continue
            # choose a random avaialble height
            index = np.random.randint(0, available_space_below.size)
            height = available_space_below[index] + 1
            # print("height: ",height)

            # this includes the 'column below' itself - for unit-width rectangles
            rows_right = self.grid[first_point[0]:first_point[0]+height, first_point[1]:]
            # print("rows right: ", rows_right)
            other_rectangles_right = np.nonzero(np.any(rows_right ,axis=0 ))
            # print("other rect right: ",other_rectangles_right)
            if other_rectangles_right[0].size != 0:
                # only first row of free space as the second is redundant at this point
                free_space_right = rows_right[0,:other_rectangles_right[0][0]]
            else:
                free_space_right = rows_right[0]

            free_space_right = np.arange(free_space_right.size)
            available_second_dimension = self.available_dimensions[height -1,:free_space_right.size]
            available_space_right = free_space_right[available_second_dimension]

            if available_space_right.size == 0:
                attempts += 1
                # print("continue")
                continue
            #choose a random avaialble width
            index =  np.random.randint(0, available_space_right.size)
            width = available_space_right[index] + 1
            # print("width", width)

            return Rectangle(first_point[0],first_point[1], height, width)

        return None

    def add_rectangle(self,rectangle):

        fp_x = rectangle.first_point_x
        fp_y = rectangle.first_point_y
        width = rectangle.width
        height = rectangle.height

        # print(f"added: x: {fp_x}, y: {fp_y}, h: {height}, w: {width}")

        self.grid[fp_x:fp_x + height, fp_y:fp_y + width] = self.rectangle_id
        self.candidate_places = self._create_list_of_candidate_plc()
        self.rectangle_area_list.append(rectangle.area)
        self.rectangle_area_list.sort()
        self.rectangle_id += 1
        self.available_dimensions[width - 1, height -1] = False
        self.available_dimensions[height -1, width - 1] = False


    def get_potential_mond_score(self, area):
        smallest = self.rectangle_area_list[0]
        biggest = self.rectangle_area_list[-1]
        if area < smallest:
            return biggest - area
        elif area > biggest:
            return  area - smallest
        else:
            return self.get_mondrian_score()


    # def create_plot(self):
    #     fig, ax = plt.subplots(1, 1, figsize=(self.grid.shape))
    #     return fig, ax

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
        my_cmap.set_under('w')
        ax.imshow(self.grid, vmin=.001, vmax=self.rectangle_id, cmap=my_cmap)
        plt.show()

    def get_mondrian_score(self):
        return self.rectangle_area_list[-1] - self.rectangle_area_list[0]


    def _create_list_of_candidate_plc(self):
        reversed_zeros = np.where(self.grid > 0, 0, 1)
        return np.argwhere(reversed_zeros)

    def _select_random_first_point(self):
        first_point_index = np.random.randint(0, self.candidate_places.shape[0])
        first_point = self.candidate_places[first_point_index,:]
        return (first_point[0], first_point[1])

    def _initialise_dimensions(self):
        d =  np.full((self.size, self.size), True)
        d[self.size -1, self.size -1] = False
        return d
