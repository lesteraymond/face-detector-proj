import cv2
import os
import numpy as np


class Train:
    def __init__(self):
        self.recog = cv2.face.LBPHFaceRecognizer_create()
        self.dataset_folder = "dataset"
        self.faces = []
        self.labels = []
        self.label_map = {}
        self.current_label = 0

        for folder in os.listdir(self.dataset_folder):
            # if file.endswith(".png"):
            folder_path = os.path.join(self.dataset_folder, folder)

            if os.path.isdir(folder_path):
                name = folder

                if name not in self.label_map:
                    self.label_map[name] = self.current_label
                    self.current_label += 1
                    print(f"Found: {name}")

                label = self.label_map[name]

                for file in os.listdir(folder_path):
                    if file.endswith(".png"):
                        img_path = os.path.join(folder_path, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        # resize = cv2.resize(img, (200, 200))

                        self.faces.append(img)
                        self.labels.append(label)
                        print(f"Loaded: {file}")

            # if os.path.isdir(folder_name):
            #     name = folder_name
            #     if name not in self.label_map:
            #         print(name)
            #         self.label_map[name] = self.current_label
            #         self.current_label += 1

            #     label = self.label_map[name]
            #     image_path = os.path.join(self.dataset_folder, folder_name)
            #     print(image_path)
            #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #     # resize = cv2.resize(img, (200, 200))

            # self.faces.append(img)
            # self.labels.append(label)

    def save_model(self):
        faces = np.array(self.faces)
        labels = np.array(self.labels)
        self.recog.train(faces, labels)
        self.recog.save("model.yml")
        print("Model saved!")
