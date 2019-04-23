import os
import glob

import random

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg,.. , orange01.jpg ..etc.

images = glob.glob('data/*.jpg')

# Shuffle the dataset to remove the bias - if present
random.shuffle(images)
# Creating labels. Consider apple=0 orange=1

labels = [0 if 'apple' in image else 1 for image in images]
data = zip(images, labels)


# Ratio
data_size = len(data)
split_size =  int(0.6 * data_size)


# Spliting the dataset

training_images, training_labels = zip(*data[:split_size])
testing_images, testing_labels = zip(*data[split_size:])
