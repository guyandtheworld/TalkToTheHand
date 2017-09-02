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
        label = {"TRUE":[1,0], "FALSE":[0, 1]}
        if self.file_dir == "P_TRUE" or self.file_dir == "P_FALSE":
            self.file_dir = self.file_dir.split("_")[1]
        features = np.array(self.gray_scale)
        labels = np.array([label[self.file_dir]]*len(self.gray_scale))
        return (features, labels)


class Data(object):
    def __init__(self, ttype):
        if ttype == "Train":
            self.true_case = LoadData("TRUE").getdata()
            self.false_case = LoadData("FALSE").getdata()
        elif ttype == "Predict":
            self.true_case = LoadData("P_TRUE").getdata()
            self.false_case = LoadData("P_FALSE").getdata()

    def train(self):
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

    def predict(self):
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
        self.y = tf.placeholder("float", [None, 2])

        W1 = tf.Variable(tf.truncated_normal([3596, 800], stddev=0.1))
        b1 = tf.Variable(tf.zeros([800]))
        W2 = tf.Variable(tf.truncated_normal([800, 2], stddev=0.1))
        b2 = tf.Variable(tf.zeros([2]))

        with tf.name_scope("Wx_b") as scope:
            Y1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
            self.model = tf.nn.softmax(tf.matmul(Y1, W2)+b2)

        with tf.name_scope("cost_function") as scope:
            self.cost_function = -tf.reduce_sum(self.y*tf.log(tf.clip_by_value(self.model,1e-10,1.0)))

        with tf.name_scope("scope") as scope:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)

    def train(self):
        train_data = Data("Train").train()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(10):
                avg_cost = 0.
                total_batch = 5
                for j in range(0, 367, 5):
                    xs = np.array(train_data[0][j:j+5])
                    ys = train_data[1][j:j+5]
                    sess.run(self.optimizer, feed_dict={self.x: xs, self.y: ys})
                    avg_cost += sess.run(self.cost_function, feed_dict={self.x: xs, self.y: ys})/5
                    print ("Iteration:", '%04d' % (i + 1), "cost=", "{:.9f}".format(avg_cost))

            saver = tf.train.Saver()
            saver.save(sess, 'TF_Model/cv_model', global_step=1000)
            train_data = Data("Predict").train()
            predictions = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
            print("Accuracy: ", accuracy.eval({self.x: train_data[0], self.y: train_data[1]}))

c = Classifier()
c.train()