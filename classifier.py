import cv2
import glob
import numpy as np
import tensorflow as tf

class PD(object):
    def __init__(self):
        self.gray_scale = []
        self.process()

    def process(self):
        print("Starting...")
        for img in glob.glob("data/*"):
            picture = cv2.imread(img)
            gray = np.array(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY))
            flattened = gray.flatten()
            self.gray_scale.append(flattened)

    def gd(self):
        features = np.array(self.gray_scale)
        labels = np.array([1]*len(self.gray_scale))
        return (features, labels)

class TF(object):
    def __init__(self):
        pass