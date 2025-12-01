import cv2
import os
from random import randint


class Capture:
    def __init__(self):
        self.dataset_path = "dataset/"
        self.suff = 0
        ##

    def set_frame(self, fr):
        self.frame = fr

    def set_faces(self, fc):
        self.faces = fc

    def write(self, file_name):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        x, y, w, h = self.faces[0]
        face = self.frame[y : y + h, x : x + w]
        self.suff = randint(0, 9999)
        file_name = f"{self.dataset_path}/{file_name}__{self.suff}.png"
        cv2.imwrite(file_name, face)
