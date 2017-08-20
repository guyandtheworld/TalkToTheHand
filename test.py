import numpy as np
import tensorflow as tf
import glob

filenames = []

for img in glob.glob("TRUE/*"):
    filenames.append(img)

filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_jpeg(content, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])

image_batch = tf.train.batch([resized_image], batch_size=8)

predictions = tf.equal(tf.argmax(image_batch, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})