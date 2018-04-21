import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input
import numpy as np
device = "/gpu:0"
from tensorflow.examples.tutorials.mnist import input_data

input_img = Input(shape=())
