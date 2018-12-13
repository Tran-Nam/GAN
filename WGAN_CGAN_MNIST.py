import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)


batch_size = 128
lr = 1e-3
img_dim = 784
noise_dim = 100
n_epochs = 20

G_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
# D_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
D_input = tf.placeholder(tf.float32, shape=[None, 784])
is_training = tf.placeholder(tf.bool)
