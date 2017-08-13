import cv2
import glob
import numpy as np
import random
import tensorflow as tf


class LoadData(object):
    def __init__(self, file_dir):
        self.gray_scale = []
        self.file_dir = file_dir
        self.process()

    def process(self):
        print("Starting...")
        for img in glob.glob(self.file_dir+"/*"):
            picture = cv2.imread(img)
            grayscale = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            flattened = grayscale.flatten()
            self.gray_scale.append(flattened)

    def getdata(self):
        label = {"TRUE":1, "FALSE":0}
        features = np.array(self.gray_scale)
        labels = np.array([np.array([label[self.file_dir]])]*len(self.gray_scale))
        return (features, labels)


class GetTrainingData(object):
    def __init__(self): 
        self.true_case = LoadData("TRUE").getdata()
        self.false_case = LoadData("FALSE").getdata()

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
        return (features, labels)


class Classifier(object):
    def __init__(self):
        self.learning_rate = 0.01
        self.training_iterations = 30
        self.batch_size = 10
        self.display_set = 1

        self.x = tf.placeholder("float", [None, 3596])
        self.y = tf.placeholder("float", [None, 1])

        self.W = tf.Variable(tf.zeros([3596, 1]))
        self.b = tf.Variable(tf.zeros([1]))

        with tf.name_scope("Wx_b") as scope:
            self.model = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        with tf.name_scope("cost_function") as scope:
            self.cost_function = -tf.reduce_sum(self.y*tf.log(self.model))

        with tf.name_scope("scope") as scope:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)

    def train(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            instance = GetTrainingData()
            training_data = instance.mix()
            m = len(training_data[0])
            cost = 0
            for x_i, y_i in zip(training_data[0], training_data[1]):
                x_t = np.array([x_i])
                y_t = np.array([y_i])
                print(x_t.shape, y_t.shape)
                sess.run(self.optimizer, feed_dict={self.x: x_i, self.y: y_i})
                cost += sess.run(self.cost_function, feed_dict = {self.x: x_i, self.y: y_i})
                print(cost)
            # avg_cost = cost/m
            # print(avg_cost)


a = Classifier()
a.train()