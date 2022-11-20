import numpy as np

class Rectangle:

    def __init__(self, first_point_x, first_point_y, height, width):
        self.first_point_x = int(first_point_x)
        self.first_point_y = int(first_point_y)
        self.height = int(height)
        self.width = int(width)
        self.area= height*width

    def attrs(self):
        return (self.first_point_x, self.first_point_y, self.height, self.width)

    def __eq__(self, other):
        if type(self) is type(other):
            return self.attrs() == other.attrs()
        else:
            return False
    def __hash__(self):
        return hash(self.attrs())


    def to_arr(self):
        return np.array([self.first_point_x, self.first_point_y, self.height, self.width, self.area], dtype="float64")
