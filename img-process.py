import cv2
import numpy as np
import time
from classifier import Classifier

class HandRecognise(object):
    def __init__(self):
        self.classifier = Classifier()
        self.classifier.train()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 58)
        self.cap.set(4, 62)
        self.cap.set(16, 1)

    def run(self):
        count = 0
        while(1):
            _, frame = self.cap.read()
            if count%5 == 0:
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('gray', grayscale)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
            print(count)
            count+=1

hg = HandRecognise()
hg.run()
cv2.destroyAllWindows()
