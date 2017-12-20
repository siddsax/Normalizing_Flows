import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
# from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 1
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)
print(X_samples.get_shape())
# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
reconstruction_loss = tf.reduce_mean(recon_loss)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('original_out/'):
    os.makedirs('original_out/')

i = 0
distribution_0 = []
distribution_1 = []
distribution_2 = []
distribution_3 = []
distribution_4 = []
distribution_5 = []
distribution_6 = []
distribution_7 = []
distribution_8 = []
distribution_9 = []
f_recon_loss = open('vae_recun_loss_4_layes.txt','w')
for it in range(1000000):
    X_mb, Y_mb = mnist.train.next_batch(mb_size)

    _, loss, reconstruction_error = sess.run([solver, vae_loss, reconstruction_loss], feed_dict={X: X_mb})
    
    if (Y_mb[0,0] == 1):
        distribution_0.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,1] == 1):
        distribution_1.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,2] == 1):
        distribution_2.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,3] == 1):
        distribution_3.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,4] == 1):
        distribution_4.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,5] == 1):
        distribution_5.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,6] == 1):
        distribution_6.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,7] == 1):
        distribution_7.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,8] == 1):
        distribution_8.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    elif (Y_mb[0,9] == 1):
        distribution_9.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())


    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print('recon loss: {:.4}'.format(reconstruction_error))

        f_recon_loss.write(str(reconstruction_error)+' '+str(loss)+'\n')
        
        import pickle
        with open('original_samples.txt','w') as f:
            pickle.dump(distribution_0,f)
            pickle.dump(distribution_1,f)
            pickle.dump(distribution_2,f)
            pickle.dump(distribution_3,f)
            pickle.dump(distribution_4,f)
            pickle.dump(distribution_5,f)
            pickle.dump(distribution_6,f)
            pickle.dump(distribution_7,f)
            pickle.dump(distribution_8,f)
            pickle.dump(distribution_9,f)

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
        fig = plot(samples)
        plt.savefig('original_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

# import pickle
# with open('original_samples.txt','w') as f:
#     pickle.dump(distribution,f)

# f.close()

# import pickle
# with open('original_samples.txt','r') as f:
#     data = pickle.load(f)


# import numpy as np
# from scipy.stats import gaussian_kde

# import matplotlib.pyplot as plt
# from matplotlib import *
# data = distribution
# x= []
# y= []
# len(data)
# for i in range(0,len(data)):
#     x.append(data[i][0][0])
#     y.append(data[i][0][1])

# len(x)
# len(y)
# # Calculate the point density
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)

# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# plt.show()
