from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import time

import tensorflow as tf
from lstm_cell import BasicLSTMCell, DropoutMaskWrapper
from lstm_cell import orthogonal_initializer
from tensorflow.examples.tutorials.mnist import input_data
from utilities import rampup, rampdown, batch_noise, adversarial_dropout, one_drop_noise
from utilities import kl_divergence_with_logit, ce_loss, qe_loss

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'dataset/mnist', "")
tf.app.flags.DEFINE_integer('seed', 1, "seed")

tf.app.flags.DEFINE_integer('num_epochs', 250, "num_epochs")
tf.app.flags.DEFINE_integer('test_epoch', 1, "testing per epoch")

tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial learning rate")
tf.app.flags.DEFINE_float('epsilon', 1e-4, ".......")
tf.app.flags.DEFINE_integer('batch_size', 100, "batch-size")
tf.app.flags.DEFINE_float('noise', None, ".......")
tf.app.flags.DEFINE_integer('num_hidden', 100, "hidden-size")

tf.app.flags.DEFINE_string('method', "adv", "")
tf.app.flags.DEFINE_string('adv_cost', 'kl', "")  # ce, qe, KL
tf.app.flags.DEFINE_integer('num_adv_change', 3, "....")
tf.app.flags.DEFINE_integer('K', 2, "....")
tf.app.flags.DEFINE_boolean('adv_onestep_random', True, "")

tf.app.flags.DEFINE_float('drop_p', 0.00, ".......")  # 0.10

tf.app.flags.DEFINE_string('init', 'ortho', "")
tf.app.flags.DEFINE_boolean('permuted', True, "")


def model(x, lstm_W, lstm_b, W, b, state_mask):
    network = BasicLSTMCell(FLAGS.num_hidden, lstm_W, lstm_b)
    network = DropoutMaskWrapper(network, state_mask, dtype=tf.float32)
    output, _ = tf.nn.static_rnn(network, x, dtype=tf.float32)
    final_hidden = output[-1]
    pyx = tf.matmul(final_hidden, W) + b
    return pyx

def upper_model(W, b, final_input, cur_adv_mask):
    return tf.matmul(final_input * cur_adv_mask, W) + b

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 784)  # 28x28x1 input img
teX = teX.reshape(-1, 784)  # 28x28x1 input img

if FLAGS.permuted:
    np.random.seed(10)
    perm = np.random.permutation(784)
    trX = trX[:, perm]
    teX = teX[:, perm]

tf.reset_default_graph()
tf.set_random_seed(FLAGS.seed)

# define X and Y
X = tf.placeholder("float32", [None, 784])
Y = tf.placeholder("float32", [None, 10])
x = tf.split(tf.reshape(X, [-1, 784]), 784, 1)

# define learning parameters
p_keep_conv = tf.placeholder("float")
w_coeff = tf.placeholder("float")
lr = tf.placeholder("float")

# define masks
temp = tf.matmul(x[0], tf.ones([1, 100]))
mask_shape = tf.shape(tf.ones_like(temp))

if FLAGS.method is not 'adv':
    state_mask = batch_noise(mask_shape, inner_seed=FLAGS.seed, keep_prob=p_keep_conv)
else:
    state_mask = tf.ones_like(temp, tf.float32)

# lstm_parameters
lstm_W_x = tf.get_variable('lstm_W_x',
                           [1, 4 * FLAGS.num_hidden],
                           initializer=orthogonal_initializer(), trainable=True)
lstm_W_h = tf.get_variable('lstm_W_h',
                           [FLAGS.num_hidden, 4 * FLAGS.num_hidden],
                           initializer=orthogonal_initializer(), trainable=True)
lstm_W = tf.concat([lstm_W_x, lstm_W_h], axis=0)
lstm_bias = tf.get_variable('lstm_bias', [4 * FLAGS.num_hidden], trainable=True)

# prediction parameters
W = tf.get_variable('W', [FLAGS.num_hidden, 10], initializer=orthogonal_initializer(), trainable=True)
b = tf.get_variable('b', [10], trainable=True)

py_x = model(x, lstm_W, lstm_bias, W, b, state_mask)
pure_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

predict_op = tf.argmax(py_x, 1)


def adv_cost_function(adv_py_x, py_x, Y):
    if FLAGS.adv_cost == 'ce':
        adv_cost = ce_loss(adv_py_x, Y)
    elif FLAGS.adv_cost == 'qe':
        adv_cost = qe_loss(adv_py_x, py_x)
    elif FLAGS.adv_cost == 'kl':
        adv_cost = kl_divergence_with_logit(adv_py_x, py_x)
    return adv_cost


if FLAGS.adv_onestep_random:
    adv_state_mask = one_drop_noise(mask_shape, inner_seed=FLAGS.seed)
    num_adv_change = FLAGS.num_adv_change - 1
else:
    adv_state_mask = tf.ones_like(state_mask)
    num_adv_change = FLAGS.num_adv_change

if FLAGS.method == 'adv':
    
    step_change = int(num_adv_change/FLAGS.K)
    for k in range(0, FLAGS.K):
        if k == (FLAGS.K-1): step_change = int(num_adv_change - k*step_change)
        adv_py_x = model(x, lstm_W, lstm_bias, W, b, adv_state_mask)
        adv_cost = adv_cost_function(adv_py_x, py_x, Y)
        Jacob = tf.stop_gradient(tf.gradients(adv_cost, [adv_state_mask], aggregation_method=2)[0])
        adv_state_mask = tf.stop_gradient( adversarial_dropout(adv_state_mask, Jacob, step_change + 1) )

    adv_py_x = model(x, lstm_W, lstm_bias, W, b, adv_state_mask)
    adv_cost = adv_cost_function(adv_py_x, py_x, Y)
    cost = pure_cost + w_coeff * adv_cost
else:
    cost = pure_cost

optimizer = tf.train.RMSPropOptimizer(lr, 0.5)
tvars = tf.trainable_variables()
gvs = optimizer.compute_gradients(cost, tvars)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), tvar) for grad, tvar in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

print("Training...")
# Launch the graph in a session
lamb = 1.0
drop_p = 1.0 - FLAGS.drop_p
learning_rate = FLAGS.learning_rate

init_op = tf.initialize_all_variables()

config = tf.ConfigProto()
with tf.Session(config=config) as sess:
    # you need to initialize all variables
    sess.run(init_op)

    training_batch = zip(range(0, len(trX), FLAGS.batch_size), range(FLAGS.batch_size, len(trX) + 1, FLAGS.batch_size))
    test_batch = zip(range(0, len(teX), FLAGS.batch_size * 10),
                     range(FLAGS.batch_size * 10, len(teX) + 1, FLAGS.batch_size * 10))


    for i in range(FLAGS.num_epochs):
        cur_w = lamb * rampup(i, 50)
        cur_lr = learning_rate * rampdown(i, 50, FLAGS.num_epochs)

        tr_acc = 0
        total_tr_cost = 0
        st = time.time()

        for start, end in training_batch:
            _, p_y, tr_cost = sess.run([train_op, predict_op, cost],
                                              feed_dict={X: trX[start:end], Y: trY[start:end],
                                                         p_keep_conv: drop_p, w_coeff: cur_w, lr: cur_lr})
            tr_acc += np.sum(np.argmax(trY[start:end], axis=1) == p_y)
            total_tr_cost += tr_cost

        ed = time.time()
        tr_acc = tr_acc / len(trX)

        if (i+1) % FLAGS.test_epoch != 0:
            print('Epoch:{0:3d}| Loss:{1:0.6f}| Training accuracy:{2:0.6f}| Time:{3:0.6f}(s)'.format(i+1, total_tr_cost, tr_acc, ed - st))
            continue

        te_acc = 0

        for start, end in test_batch:
            te_p_y = sess.run([predict_op], feed_dict={X: teX[start:end], Y: teY[start:end], p_keep_conv: 1.0})
            te_acc += np.sum(np.argmax(teY[start:end], axis=1) == te_p_y)

        te_acc = te_acc / len(teX)

        ed = time.time()

        print('Epoch:{0:3d}| Loss:{1:0.6f}| Training accuracy:{2:0.6f}| Test accuracy:{3:0.6f} | Time:{4:0.3f}(s)'.format(i+1, total_tr_cost, tr_acc, te_acc, ed - st))
