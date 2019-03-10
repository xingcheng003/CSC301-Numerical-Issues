import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import conv_layer, conv_layer2, max_pool_2x2, full_layer
from prettytable import PrettyTable

DATA_PATH = "/home/csc59866/proj1/cifar-10-batches-py"
BATCH_SIZE = 50
STEPS = 25000
lr = 1e-3
# using different learning rate and batch size
BATCH_SIZE_LIST = [25, 50, 75, 100]
LR_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
TABLE = PrettyTable()
T_TABLE = PrettyTable() #table for time

TABLE.field_names = ['Learning Rate', '25' ,'50', '75', '100']
T_TABLE.field_names = ['Learning Rate', '25' ,'50', '75', '100']

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle.Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


class CifarLoader(object):
    """
    Load and mange the CIFAR dataset.
    (for any practical use there is no reason not to use the built-in dataset handler instead)
    """
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
            .astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], \
               self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)])\
            .load()
        self.test = CifarLoader(["test_batch"]).load()


def run_simple_net(bs, lr):
    # load the data
    cifar = CifarDataManager()
    # init variables
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    # 2 conv and 1 max pooling
    conv1 = conv_layer(x, shape=[3, 3, 3, 64])
    conv2 = conv_layer(conv1, shape=[3, 3, 64, 64])
    conv2_pool = max_pool_2x2(conv2)
    # 2 conv and 1 max pooling
    conv3 = conv_layer(conv2_pool, shape=[3, 3, 64, 128])
    conv4 = conv_layer(conv3, shape=[3, 3, 128, 128])
    conv4_pool = max_pool_2x2(conv4)
    # flatten and drop to prevent overfitting
    conv4_flat = tf.reshape(conv4_pool, [-1, 8 * 8 * 128])
    conv4_drop = tf.nn.dropout(conv4_flat, keep_prob=keep_prob)
    # fully connected nn using relu as activation function
    full_0 = tf.nn.relu(full_layer(conv4_drop, 512))
    full0_drop = tf.nn.dropout(full_0, keep_prob=keep_prob)
    full_1 = tf.nn.relu(full_layer(full0_drop, 512))
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)
    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                           labels=y_))
    # original: train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    # for the table
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        print("Accuracy: {:.4}%".format(acc * 100))
        return acc*100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = cifar.train.next_batch(bs)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            '''if i % 500 == 0:
                # print(i//500)
                test(sess)'''
        result = test(sess)
    return result

def create_cifar_image():
    d = CifarDataManager()
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    images = d.train.images
    #display_cifar(images, 10)


if __name__ == "__main__":
    create_cifar_image()
    # start_time = time.time()
    for i in range(4):
        row = [LR_LIST[i]]
        t_row = [LR_LIST[i]]
        for j in range(4):
            start_time = time.time()
            result = run_simple_net(BATCH_SIZE_LIST[j], LR_LIST[i])
            row.append(result)
            t_row.append(time.time() - start_time)
        TABLE.add_row(row)
        T_TABLE.add_row(t_row)
    print(TABLE)
    print(T_TABLE)
    # run_simple_net()
    # print("--- %s seconds ---" % (time.time() - start_time))  
