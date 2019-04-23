import glob
import os
import tensorflow as tf

reader = tf.TFRecordReader()
filenames = glob.glob('*.tfrecords')

filename_queue = tf.train.string_input_producer(filenames)

_, serialized_example = reader.read(filename_queue)

features = {
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
    }

features = tf.parse_single_example(serialized_example, features=features)

image = features['image']
label = features['label']

with tf.Session() as sess:
    sess.run([image, lable])
