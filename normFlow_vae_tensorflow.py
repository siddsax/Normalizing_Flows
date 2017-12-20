import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import cPickle as pickle
# from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )

if len(sys.argv) < 3:
    print "Incorrect no. of arguments"
    print "Usage : python normFlow_vae_tensorflow.py plot_or_not num_flows"
    sys.exit()

# Read the bool to plot the graph
plot_graph = sys.argv[1]
num_flows = int(sys.argv[2])

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
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



u = []
w = []
b = []
uw = []
muw = []
u_hat = []
zwb= []
f_z = []
psi = []
psi_u = []
# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
logdet_jacobian = 0

for i in range(4):
    u.append(tf.Variable(xavier_init([z_dim,1]),name=("U_"+str(i))))
    w.append(tf.Variable(xavier_init([z_dim,1]),name=("V_"+str(i))))
    b.append(tf.Variable(xavier_init([1,1]))) #scalar
    uw.append(tf.matmul(tf.transpose(w[i]),u[i]))
    
    muw.append(-1 + tf.nn.softplus(uw[i])) # = -1 + T.log(1 + T.exp(uw))
    u_hat.append(u[i] + (muw[i] - uw[i]) * w[i] / tf.reduce_sum(tf.matmul(tf.transpose(w[i]),w[i])))
    if(i==0):
        zwb.append(tf.matmul(z_sample,w[i]) + b[i])
        f_z.append(z_sample + tf.multiply( tf.transpose(u_hat[i]), tf.tanh(zwb[i])))
    else:
        zwb.append(tf.matmul(f_z[i-1],w[i]) + b[i])
        f_z.append(f_z[i-1] + tf.multiply( tf.transpose(u_hat[i]), tf.tanh(zwb[i])))

    psi.append(tf.matmul(w[i],tf.transpose(1-tf.multiply(tf.tanh(zwb[i]), tf.tanh(zwb[i]))))) # tanh(x)dx = 1 - tanh(x)**2
    psi_u.append(tf.matmul(tf.transpose(psi[i]), u_hat[i]))
    logdet_jacobian += tf.log(tf.abs(1 + psi_u[i]))

##################################################################################

_, logits = P(f_z[-1])  # add flows thing in P


X_samples, _ = P(z)

# E[log P(X|z_k)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
reconstruction_loss = tf.reduce_mean(recon_loss)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
 # VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss - logdet_jacobian)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

# sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

p = 0


distribution = {}
for i in range(10):
    distribution[i] = []

distribution_all = []
distribution_all_without_trans = []
f = []
f_recon_loss = open('recun_loss_'+str(i) + '_layes.txt','w')
for it in range(10000):
    X_mb, Y_mb = mnist.train.next_batch(mb_size)

    _, loss, reconstruction_error = sess.run([solver, vae_loss, reconstruction_loss], feed_dict={X: X_mb})
    
    distribution_all.append(sess.run(f_z[-1],feed_dict={X: X_mb}).tolist())
    distribution_all_without_trans.append(sess.run(z_sample,feed_dict={X: X_mb}).tolist())

    for i in range(10):
        if(Y_mb[0,i]==1):
            k = i
            break
    distribution[k].append(sess.run(f_z[-1],feed_dict={X: X_mb}).tolist())

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'.format(loss))
        print('recon loss: {:.4}'.format(reconstruction_error))
        f_recon_loss.write(str(reconstruction_error)+' '+str(loss)+'\n')
        
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        p += 1
        plt.close(fig)


for i in range(10):
    with open('flow_samples_'+ str(i) + '.txt','wb') as f:
        pickle.dump(distribution[i],f)

with open('flow_samples_all.txt','wb') as f:
    pickle.dump(distribution_all,f)
    
if plot_graph:
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    markers = ['v','^','d','_','|','s','8','s','p','*']


    filname = 'flow_samples_all.txt'
    with open(filname,'rb') as f:
        data = pickle.load(f)
    x = []
    y = []
    # print(len(data))
    for j in range(0,len(data[1:50000])):
        x.append(data[j][0][0])
        y.append(data[j][0][1])
        # print(j)
    # print(data)
    # len(y)
    xy =np.vstack([x,y])
    z = (gaussian_kde(xy)(xy))
    # x_m = sum(x) / float(len(x))
    # y_m = sum(y) / float(len(x))
    ax.scatter(x, y, c=z, s=10, edgecolor='')

    print("main_done")
    for i in range(0,10):
        filname = 'flow_samples_'+ str(i) + '.txt'
        with open(filname,'r') as f:
            data = pickle.loads(f.read())
        x = []
        y = []
        # print(len(data))
        for j in range(0,len(data[1:5000])):
            x.append(data[j][0][0])
            y.append(data[j][0][1])
        # print(data)
        # len(y)
        # xy =np.vstack([x,y])
        # z = (gaussian_kde(xy)(xy))*(i+.1)/10.0
        x_m = sum(x) / float(len(x))
        y_m = sum(y) / float(len(x))
        ax.scatter(x_m, y_m, c=1000 ,s=100, edgecolor='',marker = markers[i])
        print(i)
    plt.savefig('plot' + '.png')

