import cv2
import numpy as np

class HandRecognise(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while(1):
            _, frame = self.cap.read()
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(grayscale)
            cv2.imshow('gray', grayscale)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

hg = HandRecognise()
hg.run()
cv2.destroyAllWindows()
