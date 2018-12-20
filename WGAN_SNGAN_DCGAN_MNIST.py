import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)


batch_size = 64
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

def conv2d(input, kernel, strides=[1, 2, 2, 1], scope_name='conv2d', conv_type='SAME', spectral_norm=True, update_collection=None):
    out_dim = kernel[3]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', out_dim, tf.float32, initializer=tf.constant_initializer(0))
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection)

        conv = tf.nn.bias_add(tf.nn.conv2d(input, weights, strides=strides, padding=conv_type), bias)
        return conv 

def deconv2d(input, kernel, output_shape, strides=[1, 2, 2, 1], scope_name='deconv2d', deconv_type='SAME'):
    out_dim = kernel[2]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', out_dim, tf.float32, initializer=tf.constant_initializer(0))
        try: 
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, weights, output_shape, strides=strides, padding=deconv_type), bias)
        except:
            deconv = tf.nn.bias_add(tf.nn.deconv2d(input, weights, output_shape, strides=strides, padding=deconv_type), bias)
        
        return deconv

def linear(input, output_size, scope_name='linear', spectral_norm=True, update_collection=None):
    shape = input.get_shape()
    input_size = shape[1]

    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_size, output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))

        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection)
        output = tf.add(tf.matmul(input, weights), bias)
        return output

def l2_norm(input, eps=1e-12):
    input_norm = input / (tf.reduce_sum(input**2)**0.5+eps)
    return input_norm

def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        
        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])

        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
        
        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1
        
        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection=='NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            w_norm = tf.reshape(w_mat, w_shape)

        return w_norm

def GNet(Z, batch_size, reuse=False):
    with tf.variable_scope('GNet', reuse=reuse):
        h1 = linear(Z, 7*7*128, scope_name='g_1')
        h1 = tf.layers.batch_normalization(h1, training=is_training)
        h1 = tf.nn.relu(h1)
        h1 = tf.reshape(h1, shape=[-1, 7, 7, 128])

        h2 = deconv2d(h1, kernel=[5, 5, 64, 128], output_shape=[batch_size, 14, 14, 64], scope_name='g_2')
        h2 = tf.layers.batch_normalization(h2, training=is_training)
        h2 = tf.nn.relu(h2)

        h3 = deconv2d(h2, kernel=[5, 5, 1, 64], output_shape=[batch_size, 28, 28, 1], scope_name='g_3')
        h3 = tf.nn.tanh(h3)
        h3 = tf.reshape(h3, [-1, 784])

        return h3

def DNet(X, reuse=False):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('DNet', reuse=reuse):
        h1 = conv2d(X, kernel=[5, 5, 1, 64], scope_name='d_0')
        h1 = tf.layers.batch_normalization(h1, training=is_training)
        h1 = tf.nn.leaky_relu(h1)

        h2 = conv2d(h1, kernel=[5, 5, 64, 128], scope_name='d_1')
        h2 = tf.layers.batch_normalization(h2, training=is_training)
        h2 = tf.nn.leaky_relu(h2)
        h2 = tf.reshape(h2, shape=[-1, 7*7*128])

        h3 = linear(h2, output_size=1024, scope_name='d_2')
        h3 = tf.nn.leaky_relu(h3)
        h3 = linear(h3, output_size=2, scope_name='d_3')

        return h3

G_sample = GNet(G_input, batch_size=batch_size)
# G_sample_ = GNet(G_input, batch_size=n)
print(G_sample.get_shape())

r_logit = DNet(D_input)
f_logit = DNet(G_sample, reuse=True)

D_loss = -(tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit))
G_loss = -tf.reduce_mean(f_logit)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

D_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=d_vars)
G_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(G_loss, var_list=g_vars)
clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in d_vars]

init = tf.global_variables_initializer()
D_loss_ = []
G_loss_ = []



with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/SNGAN', tf.get_default_graph())

    for i in range(5000):
        X_batch, _ = mnist.train.next_batch(batch_size)
        X_batch = X_batch*2 - 1
        Z_batch = sample_Z(batch_size, noise_dim)

        _, loss_D, _ = sess.run([D_step, D_loss, clip_D], feed_dict={D_input: X_batch, G_input: Z_batch, is_training: True})

        # Z_batch = sample_Z(batch_size, noise_dim)
        _, loss_G = sess.run([G_step, G_loss], feed_dict={G_input: Z_batch, is_training: True})
        _, loss_G = sess.run([G_step, G_loss], feed_dict={G_input: Z_batch, is_training: True})

        if(i%100==0):
            print("Step: %d\tLoss_G: %.4f\tLoss_D: %.4f"%(i, loss_G, loss_D))
            # print("Step: %d\tLoss_G: %.4f\tLoss_D: %.4f"%(i, loss_G, loss_D))
        
        if(i%10==0):
            D_loss_.append(loss_D)
            G_loss_.append(loss_G)

    # batch_size = 6
        if(i%10==0):
            n = batch_size
            canvas = np.empty((28*n, 28*n))
            for k in range(n):
                z_ = sample_Z(n, noise_dim)
                g = sess.run(G_sample, feed_dict={G_input: z_, is_training: False})
                g = (g+1)/2
                # g = -1*(g-1)
                for j in range(n):
                    canvas[k*28: (k+1)*28, j*28: (j+1)*28] = g[j].reshape([28, 28])

            plt.figure(figsize=(n, n))
            plt.imshow(canvas, origin="upper", cmap="gray")
            # plt.savefig('./image_WGAN/2_WGAN_DCGAN_MNIST_step_{}.png'.format(i))
            # plt.close()
            plt.show()

    xplot = np.arange(500)
    plt.plot(xplot, D_loss_, label='D_loss')
    plt.plot(xplot, G_loss_, label='G_loss')
    plt.legend()
    # plt.savefig('loss_curve_WGAN_DCGAN_MNIST_3.png')
    plt.show()





        
