import numpy as np


class OptimalParams(object):
    """
    This class purpose is to gather all the info relevant to skeletons pair
    """
    h = None
    w = None

    def __init__(self, first_image_parts, second_image_parts, translate, scale):
        """
        Constructor
        """
        self._skeleton_image = None
        self._has_skeleton = False
        self._rmse = None
        self._first_image_parts = first_image_parts
        self._second_image_parts = second_image_parts
        self._score = 0
        self._translate = translate
        self._scale = scale
        self._upper = None
        self._bottom = None

    @property
    def scale(self):
        return self._scale

    @property
    def translate(self):
        return self._translate

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        self._upper = value

    @property
    def bottom(self):
        return self._bottom

    @bottom.setter
    def bottom(self, value):
        self._bottom = value

    @property
    def score(self):
        return self._score

    @property
    def skeleton_image(self):
        return self._skeleton_image

    @skeleton_image.setter
    def skeleton_image(self, value):
        self.h, self.w, _ = value.shape
        self._skeleton_image = value

    @property
    def has_skeleton(self):
        return self._has_skeleton

    @has_skeleton.setter
    def has_skeleton(self, value):
        self._has_skeleton = value

    @property
    def rmse(self):
        return self._rmse

    def calculate_rmse(self):
        """
        Calculate Root Mean Square Error between skeletong upper points
        :return:
        """
        distances = []
        # top parts only (upper)
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]:
            x1 = None
            y1 = None
            x2 = None
            y2 = None
            if len(self._first_image_parts) == 1 and self._first_image_parts[0].body_parts.__contains__(i):
                x1 = self._first_image_parts[0].body_parts[i].x * self.h
                y1 = self._first_image_parts[0].body_parts[i].y * self.w
            if len(self._second_image_parts) == 1 and self._second_image_parts[0].body_parts.__contains__(i):
                x2 = self._second_image_parts[0].body_parts[i].x * self.h
                y2 = self._second_image_parts[0].body_parts[i].y * self.w
            if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
                distances.append(self.calculateDistance([x1, y1], [x2, y2]))
        self._rmse = np.sqrt(np.array(distances).mean())

    def calculateDistance(self, point1, point2):
        """
        Calculate Euclidean Distance
        :param point1:
        :param point2:
        :return:
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_skeleton_score(self, body_parts):
        """
        Sum each point's score
        :param body_parts:
        :return:
        """
        for i in range(0, 17):
            if len(self._first_image_parts) == 1 and self._first_image_parts[0].body_parts.__contains__(i):
                self._score += body_parts[0].body_parts[i].score

