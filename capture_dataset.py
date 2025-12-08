import cv2
import os
from random import randint


class Capture:
    def __init__(self):
        self.dataset_path = "dataset"
        self.suff = 0
        ##

    # def set_frame(self, fr):
    #     self.frame = fr

    def set_face(self, fc):
        self.face = fc

    def write(self, folder_name):
        if not os.path.exists(self.dataset_path):
            # print("alyanna")
            os.makedirs(self.dataset_path)

        if not os.path.exists(f"{self.dataset_path}/{folder_name}"):
            # print("hanni")
            os.makedirs(f"{self.dataset_path}/{folder_name}")

        self.suff = randint(0, 1000000)
        file_name = f"{self.dataset_path}/{folder_name}/f__{self.suff}.png"
        resize = cv2.resize(self.face, (200, 200))
        cv2.imwrite(file_name, resize)
        print(f"{file_name} Saved!")
