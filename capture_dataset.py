import cv2


class Capture:
    def __init__(self):
        self.dataset_path = "dataset/"
        self.pref = 1

    def set_frame(self, fr):
        self.frame = fr

    def set_faces(self, fc):
        self.faces = fc

    def write(self):
        x, y, w, h = self.faces[0]
        face = self.frame[y : y + h, x : x + w]
        file_name = f"{self.dataset_path}/__{self.pref}.png"
        cv2.imwrite(file_name, face)
        self.pref = self.pref + 1
