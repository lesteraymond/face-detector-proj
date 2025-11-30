import os
import cv2
import numpy as np


class Train:
    def __init__(self):
        self.recog = cv2.face.LBPHFaceRecognizer_create()

        self.faces = []
        self.labels = []
        self.label_d = {}
        self.current_label = 0

        for file_name in os.listdir("dataset/"):
            file_path = os.path.join("dataset/", file_name)
            # print(file_path)
            # print(os.path.isdir(file_path))

            self.label_d[self.current_label] = file_name
            # print(image_path)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            self.faces.append(img)
            self.labels.append(self.current_label)
            self.current_label += 1

    def save_model(self):
        faces = np.array(self.faces)
        labels = np.array(self.labels)

        self.recog.train(faces, labels)
        self.recog.save("model.py")
