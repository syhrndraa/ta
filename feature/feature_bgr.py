import cv2
import numpy


class BGR:
    def feature(filename):        
        avg_color_per_row = numpy.average(filename, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        return avg_color