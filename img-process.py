import cv2
import numpy as np

class HandRecognise(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while(1):
            _, frame = self.cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower = np.array([0, 10, 60], dtype = "uint8") 
            upper = np.array([20, 150, 255], dtype = "uint8")

            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(frame,frame, mask= mask)

            cv2.imshow('frame',frame)
            cv2.imshow('mask',mask)
            cv2.imshow('res',res)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

hg = HandRecognise()
hg.run()
cv2.destroyAllWindows()
