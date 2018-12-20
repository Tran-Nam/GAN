import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import time
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
# is_training = tf.placeholder(tf.bool)

def leaky_relu(x, alpha=0.2):
    return 0.5*(1+alpha)*x + 0.5*(1-alpha)*abs(x)

def sample_Z(n_sample, n_dim):
    return np.random.uniform(-1., 1., size=[n_sample, n_dim])

def GNet(Z, reuse=False):
    with tf.variable_scope("GNet", reuse=reuse):
        h1 = tf.layers.dense(Z, units=7*7*128)
        # h1 = tf.layers.batch_normalization(h1, training=is_training)
        h1 = tf.nn.relu(h1)
        h1 = tf.reshape(h1, shape=[-1, 7, 7, 128])

        h2 = tf.layers.conv2d_transpose(h1, filters=64, kernel_size=5, strides=2, padding='same')
        # h2 = tf.layers.batch_normalization(h2, training=is_training)
        h2 = tf.nn.relu(h2)

        h3 = tf.layers.conv2d_transpose(h2, filters=1, kernel_size=5, strides=2, padding='same')
        h3 = tf.nn.tanh(h3)
        h3 = tf.reshape(h3, [-1, 784])
        return h3

def DNet(X, reuse=False):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope("DNet", reuse=reuse):
        h1 = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=2, padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_training)
        h1 = tf.nn.leaky_relu(h1)
        # h1 = leaky_relu(h1)

        h2 = tf.layers.conv2d(h1, filters=128, kernel_size=5, strides=2, padding='same')
        # h2 = tf.layers.batch_normalization(h2, training=is_training)
        h2 = tf.nn.leaky_relu(h2)
        # h2 = leaky_relu(h2)
        h2 = tf.reshape(h2, shape=[-1, 7*7*128])

        h3 = tf.layers.dense(h2, 1024)
        # h3 = tf.layers.batch_normalization(h3, training=is_training)
        h3 = tf.nn.leaky_relu(h3)
        # h3 = leaky_relu(h3)

        out = tf.layers.dense(h3, 2) # 1 ? 2
        return out 

G_sample = GNet(G_input)
print(G_sample.get_shape())
r_img = DNet(D_input)
f_img = DNet(G_sample, reuse=True)

# D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_img, labels=tf.ones_like(r_img))
#         + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_img, labels=tf.zeros_like(f_img)))
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_img, labels=tf.ones_like(f_img)))
D_loss = -(tf.reduce_mean(r_img) - tf.reduce_mean(f_img))
G_loss = -tf.reduce_mean(f_img)
# GP
fake_data = G_sample 
real_data = D_input
alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
x_hat = alpha*real_data + (1-alpha)*fake_data
D_x_hat = DNet(x_hat, reuse=True)
grad_D_x_hat = tf.gradients(D_x_hat, [x_hat])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
D_loss +=  10*gradient_penalty

z_sum = tf.summary.histogram('z', G_input)
D_loss_sum = tf.summary.scalar('D_loss', D_loss)
G_loss_sum = tf.summary.scalar('G_loss', G_loss)
GP_1_sum = tf.summary.scalar('GP', gradient_penalty)
D_sum = tf.summary.merge([z_sum, D_loss_sum])
G_sum = tf.summary.merge([z_sum, G_loss_sum])
# GP_sum = tf.summary.merge([z_sum, GP_1_sum])

D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNet")
G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GNet")

D_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=D_vars)
G_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars)
# clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in D_vars]

init = tf.global_variables_initializer()
D_loss_ = []
G_loss_ = []

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/WGAN_GP_1/', tf.get_default_graph())

    for i in range(10000):
        start = time.time()
        X_batch, _ = mnist.train.next_batch(batch_size)
        # X_batch = np.reshape(X_batch, newshape=[-1, 28, 28, 1])
        X_batch = X_batch * 2. - 1
        Z_batch = sample_Z(batch_size, noise_dim)

        _, loss_D, D_sum_ = sess.run([D_step, D_loss, D_sum], feed_dict={D_input: X_batch, G_input: Z_batch})
        writer.add_summary(D_sum_, i)

        if i%5==0:
            Z_batch = sample_Z(batch_size, noise_dim)
            _, loss_G, G_sum_ = sess.run([G_step, G_loss, G_sum], feed_dict={G_input: Z_batch})
            writer.add_summary(G_sum_, i)
            # writer.add_summary(GP_sum_, i)


        if i%100==0:
            print("Step: %d\tLoss_G: %.4f\tLoss_D: %.4f"%(i, loss_G, loss_D))
            # print('Time: %.2f'%(time.time() - start))

        if i%100==0:
            D_loss_.append(loss_D)
            G_loss_.append(loss_G)

        # if i%5000==0:
    # plt.figure(figsize=(100, 100))
    n = 32
    canvas = np.empty((28*n, 28*n))
    for k in range(n):
        z_ = sample_Z(n, noise_dim)
        g = sess.run(G_sample, feed_dict={G_input: z_})
        g = (g+1)/2
        # g = -1*(g-1)
        for j in range(n):
            canvas[k*28: (k+1)*28, j*28: (j+1)*28] = g[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.savefig('./image_WGAN/11_WGAN_GP_DCGAN_MNIST_step_{}.png'.format(i))
    # plt.close()
    plt.show()
        

    xplot = np.arange(100)
    plt.plot(xplot, D_loss_, label='D_loss')
    plt.plot(xplot, G_loss_, label='G_loss')
    plt.legend()
    # plt.savefig('loss_curve_WGAN_DCGAN_MNIST_3.png')
    plt.show()
            

    