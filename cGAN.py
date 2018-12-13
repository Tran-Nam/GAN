import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)
# print(type(mnist))

# a = mnist.train.next_batch(32)
# print(a[0].shape)

hidden_unit = 512
img_dim = 784
noise_dim = 100
batch_size = 32
lr = 1e-4

def sample_Z(n_sample, n_dim):
    return np.random.uniform(-1., 1., size=[n_sample, n_dim])

def GNet(Z, y, reuse=False):
    with tf.variable_scope("GNet", reuse=reuse):
        a = tf.concat([Z, y], axis=1)
        h1 = tf.layers.dense(a, units=hidden_unit, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, units=hidden_unit, activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, units=hidden_unit, activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h3, units=img_dim)
        return out 

def DNet(X, y, reuse=False):
    with tf.variable_scope("DNet", reuse=reuse):
        a = tf.concat([X, y], axis=1)
        h1 = tf.layers.dense(a, units=hidden_unit, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, units=hidden_unit, activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, units=hidden_unit, activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h3, 1)
        return out 

G_input = tf.placeholder(tf.float32, [None, noise_dim], name="G_input")
D_input = tf.placeholder(tf.float32, [None, img_dim], name="D_input")
y = tf.placeholder(tf.float32, [None, 10], name='Label')

G_sample = GNet(G_input, y)
r_logits = DNet(D_input, y)
f_logits = DNet(G_sample, y, reuse=True)

D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits))
                    + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))



D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNet")
G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GNet")

D_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=D_vars)
G_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars)

init = tf.global_variables_initializer()
onehot = np.eye(10)

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/GAN_MNIST', tf.get_default_graph())

    for i in range(100000):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        Z_batch = sample_Z(batch_size, noise_dim)

        _, loss_D = sess.run([D_step, D_loss], feed_dict={D_input: X_batch, G_input: Z_batch, y: y_batch}) 
        # print("---")
        # if i%5 == 0:
        y_batch = np.random.randint(0, 9, (batch_size, 1))
        y_batch = onehot[y_batch.astype(np.int32)].squeeze()
        # if i==0:
        #     print(y_batch.get_shape())
        _, loss_G = sess.run([G_step, G_loss], feed_dict={G_input: Z_batch, y: y_batch})

        if(i%5000==0):
            print("Step: %d\tG_loss: %.4f\tD_loss: %.4f"%(i, loss_G, loss_D))

    # z_ = sample_Z(6, noise_dim)
    # data_ = sess.run(G_sample, feed_dict={G_input: z_})
    # print(data_.shape)
    n = 6
    canvas = np.empty((28*n, 28*n))
    for i in range(n):
        z_ = sample_Z(n, noise_dim)
        y_batch = np.zeros((n, 1))
        y_batch[:, 0] = 8
        # y_batch[3:, 0] = 7
        # y_batch[0] = 1
        g = sess.run(G_sample, feed_dict={G_input: z_, y: onehot[y_batch.astype(np.int32)].squeeze()})
        # g = -1 * (g-1)
        for j in range(n):
            canvas[i*28: (i+1)*28, j*28: (j+1)*28] = g[j].reshape([28, 28])
    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()


    # plt.imshow(data_.reshape(28, 28).T, cmap=plt.cm.gray)
    # plt.show()
    