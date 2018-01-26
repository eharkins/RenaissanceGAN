'''
An example of distribution approximation using Generative Adversarial Networks
in TensorFlow.

Based on the blog post by Eric Jang:
http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html,

and of course the original GAN paper by Ian Goodfellow et. al.:
https://arxiv.org/abs/1406.2661.

The minibatch discrimination technique is taken from Tim Salimans et. al.:
https://arxiv.org/abs/1606.03498.
'''

import argparse
import numpy as np
import tensorflow as tf
import random
import madmom.utils.midi as md
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib import animation
import seaborn as sns
import h5py

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

DATA_SIZE = 10
DATASET_SIZE = 600

class MusicNotes(object):
    def extract_data(self, file):
        f = h5py.File(file, "r")
        a_group_key = list(f.keys())[0]
        data = np.array(list(f[a_group_key]))
        data = data.reshape(DATASET_SIZE, DATA_SIZE)
        return data
    def __init__(self, file):
        self.data = self.extract_data(file)

    def sample(self, N):
        i = random.randint(0, self.data.size)
        samples = self.data[0:N]
        # if (i < self.data.size-N):
        #     samples = self.data[i:i+N]
        # else:
        #     samples = np.concatenate(self.data[i:self.data.size], self.data[0:self.data.size-i])
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    # def sample(self, N):
    #     return np.linspace(-self.range, self.range, N) + \
    #         np.random.random(N) * 0.01

        #generates MusicNotes randomly????
    def sample(self, N):

        # i = random.randint(0, self.data.self)
        # samples = self.data[0:N]
        # if (i < self.data.size-N):
        #     samples = self.data[i:i+N]
        # else:
        #     samples = np.concatenate(self.data[i:self.data.size], self.data[0:self.data.size-i])
        arr = np.empty([N, DATA_SIZE])
        for x in range(N):
            i = random.randint(21, 33)

            vector = []
            vector.append(1.04166667/7.29166667)
            vector.append((i-21)/24)
            vector.append(1)
            vector.append(1)
            vector.append(0)

            vector.append(4.166667/7.29166667)
            vector.append((i+9)/24)
            vector.append(1)
            vector.append(1)
            vector.append(0)
            arr[x] = vector

        return np.array(arr)



def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, DATA_SIZE, 'g1')
    # print "generated: "
    # print h1
    return h1


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))

def batchToNotes(notes):
    # print(notes)
    notes[0] = notes[0]*7.29166667
    notes[1] = notes[1]*24 + 21
    notes[2] = notes[2]*2.08333333
    notes[3] = notes[3]*100
    notes[4] = notes[4]

<<<<<<< HEAD
    return np.array([notes[0:5], notes[5:10]])
<<<<<<< HEAD
=======
    notes[5] = notes[5]*7.29166667
    notes[6] = notes[6]*24 + 21
    notes[7] = notes[7]*2.08333333
    notes[8] = notes[8]*100
    notes[9] = notes[9]

    for x in range(len(notes)):
        if (notes[x] < 0 or notes[x] > 256):
            notes[x] = 0
    songNotes = np.array([notes[0:5], notes[5:10]])


    return songNotes
>>>>>>> parent of 2fa2e0d... asdf

=======
    
>>>>>>> parent of 591e1bb... midi file generated - still bug with reshaping in sampleSound


def getMidi(notesData):
    print(notesData)
    # x = tf.placeholder(tf.float32, shape = (8, 10))
    # #notes = generator.eval(( feed_dict={self.z: self.sample_z}))
    # notes = generator.eval(feed_dict={x: np.array((8,10))},  tf.Session())
    #convert to numpy array
    #convert to midi
    # print(notesData)
    for x in range(8):
        songNotes = batchToNotes(notesData[x])
        m = md.MIDIFile.from_notes(songNotes)
        print(songNotes)
        #save to file
        filename = str(x)
        m.write("song.mid")


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, DATA_SIZE))
            self.G = generator(self.z, params.hidden_size)
        # print "generator is:"
        # print self.G
        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, DATA_SIZE))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # update discriminator
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, DATA_SIZE)),
                model.z: np.reshape(z, (params.batch_size, DATA_SIZE))
            })

            # update generator
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, DATA_SIZE))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

        sampleSound(model, session, data, gen.range, params.batch_size)
            # if params.anim_path and (step % params.anim_every == 0):
            #     anim_frames.append(
            #         samples(model, session, data, gen.range, params.batch_size)
            #     )

        # if params.anim_path:
        #     save_animation(anim_frames, params.anim_path, gen.range)
        # else:
        #     samps = samples(model, session, data, gen.range, params.batch_size)
        #     plot_distributions(samps, gen.range)
        #getMidi(model.G, session)


def sampleSound(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):

    # print "batch size is: "
    # print batch_size
    # print "model is:"
    # print model
    # print "data is: "
    # print data
    # print "sample_range is: "
    # print sample_range

    '''
    Return a pg where db is the current decision
    pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    arr = np.empty([8, DATA_SIZE])
    for x in range(8):
        # i = random.randint(21, 33)
        i = x + 13
        vector = []
        vector.append(1.04166667/7.29166667)
        vector.append((i-21)/24)
        vector.append(1)
        vector.append(1)
        vector.append(0)

        vector.append(4.166667/7.29166667)
        vector.append((i+9)/24)
        vector.append(1)
        vector.append(1)
        vector.append(0)
        arr[x] = vector

    xs = zs = np.array(arr)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # # decision boundary
    # db = np.zeros((num_points, 1))
    # for i in range(num_points // batch_size):
    #     print "xs[batch_size * i:batch_size * (i + 1)] is"
    #     print (xs[batch_size * i:batch_size * (i + 1)])
    #     db[batch_size * i:batch_size * (i + 1)] = session.run(
    #         model.D1,
    #         {
    #             model.x: np.reshape(
    #                 xs[batch_size * i:batch_size * (i + 1)],
    #                 (batch_size, DATA_SIZE)
    #             )
    #         }
    #     )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    # zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, DATA_SIZE))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, DATA_SIZE)
                )
            }
        )
        getMidi(g[batch_size * i: batch_size * (i+1)])
    pg, _ = np.histogram(g, bins=bins, density=True)
    print('pg')
    print(pg)

    return pg




def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):

    # print "batch size is: "
    # print batch_size
    # print "model is:"
    # print model
    # print "data is: "
    # print data
    # print "sample_range is: "
    # print sample_range

    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    arr = np.empty([8, DATA_SIZE])
    for x in range(8):
        # i = random.randint(21, 33)
        i = x + 13
        vector = []
        vector.append(1.04166667/7.29166667)
        vector.append((i-21)/24)
        vector.append(1)
        vector.append(1)
        vector.append(0)

        vector.append(4.166667/7.29166667)
        vector.append((i+9)/24)
        vector.append(1)
        vector.append(1)
        vector.append(0)
        arr[x] = vector

    xs = np.array(arr)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        # print "xs[batch_size * i:batch_size * (i + 1)] is"
        # print (xs[batch_size * i:batch_size * (i + 1)])
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, DATA_SIZE)
                )
            }
        )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, DATA_SIZE)
                )
            }
        )
        getMidi(g[batch_size * i: batch_size * (i+1)])
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):
    model = GAN(args)
    train(model, MusicNotes("midi_data.hdf5"), GeneratorDistribution(range=8), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=500,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
