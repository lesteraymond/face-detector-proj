import cv2
import os
import numpy as np


class Train:
    def __init__(self):
        self.recog = cv2.face.LBPHFaceRecognizer_create()
        self.dataset_folder = "dataset"

        self.faces = []
        self.labels = []
        self.names = {}
        self.current_label = 0

        self.label_map = {}

        for file in os.listdir(self.dataset_folder):
            if file.endswith("png"):
                name = file.split("_")[0]

                self.label_map[name] = self.current_label
                self.current_label += 1

                label = self.label_map[name]
                image_path = os.path.join(self.dataset_folder, file)
                print(image_path)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                resize = cv2.resize(img, (200, 200))

                self.faces.append(resize)
                self.labels.append(label)

    def save_model(self):
        faces = np.array(self.faces)
        labels = np.array(self.labels)
        self.recog.train(faces, labels)
        self.recog.save("model.yml")
