import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
# from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )





mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 1
z_dim = 2
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

distribution = {}
distribution_all = []
distribution_all_without_trans = []
distribution[0] = []
distribution[1] = []
distribution[2] = []
distribution[3] = []
distribution[4] = []
distribution[5] = []
distribution[6] = []
distribution[7] = []
distribution[8] = []
distribution[9] = []
print(type(distribution[0]))
f0 = open('flow_samples_0.txt','w')
f1 = open('flow_samples_1.txt','w')
f2 = open('flow_samples_2.txt','w')
f3 = open('flow_samples_3.txt','w')
f4 = open('flow_samples_4.txt','w')
f5 = open('flow_samples_5.txt','w')
f6 = open('flow_samples_6.txt','w')
f7 = open('flow_samples_7.txt','w')
f8 = open('flow_samples_8.txt','w')
f9 = open('flow_samples_9.txt','w')

f_all = open('flow_samples_all.txt','w')
f_all_without_trans = open('flow_samples_all_without_trans.txt','w')
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

# z_mu, z_logvar = Q(X)
# z_sample = sample_z(z_mu, z*logvar)
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

# 
# =============================== TRAINING ====================================
# 
z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
u =  tf.Variable(xavier_init([z_dim,1]),name="U")
w =  tf.Variable(xavier_init([z_dim,1]),name="V")
b =  tf.Variable(xavier_init([1,1])) #scalar
uw = tf.matmul(tf.transpose(w),u)

muw = -1 + tf.nn.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
u_hat = u + (muw - uw) * w / tf.reduce_sum(tf.matmul(tf.transpose(w),w))
zwb = tf.matmul(z_sample,w) + b
f_z= z_sample + tf.multiply( tf.transpose(u_hat), tf.tanh(zwb))
psi = tf.matmul(w,tf.transpose(1-tf.multiply(tf.tanh(zwb), tf.tanh(zwb)))) # tanh(x)dx = 1 - tanh(x)**2
psi_u = tf.matmul(tf.transpose(psi), u_hat)

u_2 =  tf.Variable(xavier_init([z_dim,1]),name="U_2")
w_2 =  tf.Variable(xavier_init([z_dim,1]),name="V_2")
b_2 =  tf.Variable(xavier_init([1,1])) #scalar
uw_2 = tf.matmul(tf.transpose(w_2),u_2)

muw_2 = -1 + tf.nn.softplus(uw_2) # = -1 + T.log(1 + T.exp(uw))
u_hat_2 = u_2 + (muw_2 - uw_2) * w_2 / tf.reduce_sum(tf.matmul(tf.transpose(w_2),w_2))
zwb_2 = tf.matmul(f_z,w_2) + b_2
f_z_2= f_z + tf.multiply( tf.transpose(u_hat_2), tf.tanh(zwb_2))
psi_2 = tf.matmul(w_2,tf.transpose(1-tf.multiply(tf.tanh(zwb_2), tf.tanh(zwb_2)))) # tanh(x)dx = 1 - tanh(x)**2
psi_u_2 = tf.matmul(tf.transpose(psi_2), u_hat_2)

###############################################################################3
u_3 =  tf.Variable(xavier_init([z_dim,1]),name="U_3")
w_3 =  tf.Variable(xavier_init([z_dim,1]),name="V_3")
b_3 =  tf.Variable(xavier_init([1,1])) #scalar
uw_3 = tf.matmul(tf.transpose(w_3),u_3)

muw_3 = -1 + tf.nn.softplus(uw_3) # = -1 + T.log(1 + T.exp(uw))
u_hat_3 = u_3 + (muw_3 - uw_3) * w_3 / tf.reduce_sum(tf.matmul(tf.transpose(w_3),w_3))
zwb_3 = tf.matmul(f_z_2,w_3) + b_3
f_z_3= f_z_2 + tf.multiply( tf.transpose(u_hat_3), tf.tanh(zwb_3))
psi_3 = tf.matmul(w_3,tf.transpose(1-tf.multiply(tf.tanh(zwb_3), tf.tanh(zwb_3)))) # tanh(x)dx = 1 - tanh(x)**2
psi_u_3 = tf.matmul(tf.transpose(psi_3), u_hat_3)


####################################################################################
u_4 =  tf.Variable(xavier_init([z_dim,1]),name="U_4")
w_4 =  tf.Variable(xavier_init([z_dim,1]),name="V_4")
b_4 =  tf.Variable(xavier_init([1,1])) #scalar
uw_4 = tf.matmul(tf.transpose(w_4),u_4)

muw_4 = -1 + tf.nn.softplus(uw_4) # = -1 + T.log(1 + T.exp(uw))
u_hat_4 = u_4 + (muw_4 - uw_4) * w_4 / tf.reduce_sum(tf.matmul(tf.transpose(w_4),w_4))
zwb_4 = tf.matmul(f_z_3,w_4) + b_4
f_z_4= f_z_3 + tf.multiply( tf.transpose(u_hat_4), tf.tanh(zwb_4))
psi_4 = tf.matmul(w_4,tf.transpose(1-tf.multiply(tf.tanh(zwb_4), tf.tanh(zwb_4)))) # tanh(x)dx = 1 - tanh(x)**2
psi_u_4 = tf.matmul(tf.transpose(psi_4), u_hat_4)




logdet_jacobian = tf.log(tf.abs(1 + psi_u))+tf.log(tf.abs(1 + psi_u_2))+tf.log(tf.abs(1 + psi_u_3))+tf.log(tf.abs(1 + psi_u_4))
_, logits = P(f_z_4)  # add flows thing in P






X_samples, _ = P(z)

# E[log P(X|z_k)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
 # VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss - logdet_jacobian)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
writer = tf.summary.FileWriter("../", graph=tf.get_default_graph())
# distribution = []
for it in range(500000):
    X_mb, Y_mb = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    distribution_all.append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    distribution_all_without_trans.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())
    if ((Y_mb[0,0]) == 1):
        distribution[0].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif((Y_mb[0,1]) == 1):
        distribution[1].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,2]) == 1):
        distribution[2].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,3]) == 1):
        distribution[3].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,4]) == 1):
        distribution[4].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,5]) == 1):
        distribution[5].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,6]) == 1):
        distribution[6].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,7]) == 1):
        distribution[7].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,8]) == 1):
        distribution[8].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    elif ((Y_mb[0,9]) == 1):
        distribution[9].append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())


    # distribution.append(sess.run(f_z_4,feed_dict={X: X_mb}).tolist())
    # print(distribution)
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

import pickle
with open('samples.txt','w') as f:
    pickle.dump(distribution_all,f)

f.close()

pickle.dump(distribution[0],f0)
pickle.dump(distribution[1],f1)
pickle.dump(distribution[2],f2)
pickle.dump(distribution[3],f3)
pickle.dump(distribution[4],f4)
pickle.dump(distribution[5],f5)
pickle.dump(distribution[6],f6)
pickle.dump(distribution[7],f7)
pickle.dump(distribution[8],f8)
pickle.dump(distribution[9],f9)
pickle.dump(distribution_all,f_all)
pickle.dump(distribution_all_without_trans,f_all_without_trans)

import pickle
with open('samples.txt','r') as f:
    data = pickle.load(f)


import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
data = distribution[-10000:]
x= []
y= []
len(data)
for i in range(0,len(data)):
    x.append(data[i][0][0])
    y.append(data[i][0][1])

len(x)
len(y)
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100, edgecolor='')
plt.show()

