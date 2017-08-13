import cv2
import glob
import numpy as np
import tensorflow as tf
import random

class LoadData(object):
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

class GetTrainingData(object):
    def __init__(self): 
        self.true_case = LoadData("TRUE").gd()
        self.false_case = LoadData("FALSE").gd()

    def mix(self):
        true_case = self.true_case
        false_case = self.false_case
        features = []
        labels = []
        for i, j in zip(true_case[0], true_case[1]):
            features.append(i)
            labels.append(j)
        for i, j in zip(false_case[0], false_case[1]):
            features.append(i)
            labels.append(j)
        features = np.array(features)
        labels = np.array(labels)
        c = list(zip(features, labels))
        random.shuffle(c)
        features, labels = zip(*c)
        train = features, labels
        return train