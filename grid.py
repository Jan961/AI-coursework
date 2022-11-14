import numpy as np


class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
        self.rectangle_id = 0
        self.mondrian_score = 0
        self.candidate_places = self._create_list_of_candidate_plc()
        self.available_dimensions = np.full((size, size), True)


    def create_new_rectangle(self):

        found_rectangle = False
        while not found_rectangle:
            first_point = self.select_random_first_point()
            column_below = self.grid[first_point[0]:, first_point[1]] #this includes the point itself (for unit height rectangles)
            other_rectangles_below = np.nonzero(column_below)[0]

            if other_rectangles_below.size != 0:
                free_space_below = column_below[:other_rectangles_below[0]]
            else:
                free_space_below = column_below

            # insert indices into our column of zeros to help filter available dimensions
            free_space_below = np.arange(free_space_below.size)
            available_first_dimension = self.available_dimensions.any(axis=1)[:free_space_below.size]
            available_space_below = free_space_below[available_first_dimension]

            # find a different point if there is no available space below (taking into account that some dimensions may have
            # already been used)
            if available_space_below.size == 0:
                continue

            index = np.random.randint(0, available_space_below.size)
            height = available_space_below[index] + 1

            # this includes the 'column below' itself - for unit-width rectangles
            rows_right = self.grid[first_point[0]:first_point[0]+height-1,first_point[1]:]

            other_rectangles_right = np.nonzero(np.all(rows_right ,axis=0 ))

            if other_rectangles_right.size !=0:
                # only first row of free space as the second is redundant at this point
                free_space_right = rows_right[0,other_rectangles_right[0]]
            else:
                free_space_right = rows_right[0]

            free_space_right = np.arange(free_space_right.size)
            available_second_dimension = self.available_dimensions[height -1,:free_space_right.size]
            available_space_right = free_space_right[available_second_dimension]

            if available_space_right.size == 0:
                continue

            index =  np.random.randint(0, available_space_right.size)
            width = available_space_right[index] + 1

            found_rectangle = True

        return (first_point, width, height)


    def add_rectangle(self,first_point, width, height):
        self.rectangle_id += 1
        fp_x = first_point[0]
        fp_y = first_point[1]

        self.grid[fp_x:fp_x + height - 1, fp_y:fp_y + width - 1] = self.rectangle_id
        self.candidate_places = self._create_list_of_candidate_plc()



    def show_grid(self):
        pass

    def get_mondrian_score(self):
        pass

    def _create_list_of_candidate_plc(self):
        reversed_zeros = np.where(self.grid > 0, 0, 1)
        return np.argwhere(reversed_zeros)

    def _select_random_first_point(self):
        first_point_index = np.random.randint(0, self.candidate_places.shape[0])
        first_point = self.candidate_places[first_point_index,:]
        return first_point
