import cv2
import glob
import numpy as np
import tensorflow as tf

class PD(object):
    def __init__(self, file_dir):
        self.gray_scale = []
        self.file_dir = file_dir
        self.process()

    def process(self):
        print("Starting...")
        for img in glob.glob(self.file_dir+"/*"):
            picture = cv2.imread(img)
            flattened = picture.flatten()
            self.gray_scale.append(flattened)

    def gd(self):
        label = {"TRUE":1, "FALSE":0}
        features = np.array(self.gray_scale)
        labels = np.array([label[self.file_dir]]*len(self.gray_scale))
        return (features, labels)

class TF(object):
    def __init__(self): 
        true_case = PD("TRUE").gd()
        false_case = PD("FALSE").gd()
        self.mix(true_case, false_case)

    def mix(self, true_case, false_case):
        print(len(true_case[0]), len(false_case[0].shape))
        # for i, j in zip(true_case[0], false_case[0]):
        #     print(i, j)
        # features = np.concatenate((true_case[0], false_case[0])) 
        # labels = np.concatenate(true_case[1], false_case[1])
        # return features, labels