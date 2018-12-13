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

def leaky_relu(x, alpha=0.2):
    return 0.5*(1+alpha)*x + 0.5*(1-alpha)*abs(x)

def sample_Z(n_sample, n_dim):
    return np.random.uniform(-1., 1., size=[n_sample, n_dim])

def conv2d(img, filter, stride, )
