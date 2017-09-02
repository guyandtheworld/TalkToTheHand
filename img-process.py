import tensorflow as tf
import cv2
import numpy as np
import time

class HandRecognise(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 62)
        self.cap.set(4, 58)

    def run(self):
        count = 0
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('TF_Model/cv_model-1000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('TF_Model/'))
            graph = tf.get_default_graph()
            model = graph.get_tensor_by_name("Wx_b/Softmax:0")
            x = graph.get_tensor_by_name("Placeholder:0")
            while(1):
                _, frame = self.cap.read()
                if count%3 == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (62, 58))
                    cv2.imshow('Vision', image)
                    # TensorFlow predictions
                    flattened = image.flatten()
                    feed_dict = {x: flattened[np.newaxis]}
                    classification = sess.run(model, feed_dict)
                    print(classification)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break
                    print("Frame: " + str(int(count/3)))
                count+=1

hg = HandRecognise()
hg.run()
cv2.destroyAllWindows()
