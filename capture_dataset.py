import cv2


class Capture:
    def __init__(self):
        self.dataset_path = "dataset/"
        self.pref = 1

    def set_image(self, image):
        self.img = image

    def write(self):
        file_name = f"{self.dataset_path}/__{self.pref}.png"
        cv2.imwrite(file_name, self.img)
        self.pref = self.pref + 1
