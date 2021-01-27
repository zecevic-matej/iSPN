from observations import mnist, fashion_mnist
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
# import visdom

import os
import csv
import argparse

import RAT_SPN
from relu_mlp import ReluMLP
import region_graph
import model


deprecation._PRINT_DEPRECATION_WARNINGS = False

# vis = visdom.Visdom()
np.set_printoptions(precision=3)


def dump_attributes(obj, filename):
    w = csv.writer(open(filename, 'w'))
    for key, val in vars(obj).items():
        w.writerow([str(key), str(val)])


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def load_mnist(fashion=False):
    if fashion:
        (train_im, train_lab), (test_im, test_lab) = fashion_mnist("data/fashion")
    else:
        (train_im, train_lab), (test_im, test_lab) = mnist("data/mnist")
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)
    std_eps = 1e-7
    # train_im = (train_im - train_im_mean) / (train_im_std + std_eps)
    # test_im = (test_im - train_im_mean) / (train_im_std + std_eps)

    train_im /= 255.0
    test_im /= 255.0
    train_im = np.reshape(train_im, (-1, 28, 28, 1))
    test_im = np.reshape(test_im, (-1, 28, 28, 1))

    return (train_im, train_lab), (test_im, test_lab)



def show_graph():
    for op in tf.get_default_graph().get_operations():
        tupl = op.values()
        if len(tupl) == 0:
            continue
        tensor = tupl[0]

        size = 1
        for dim in tensor.shape:
            if dim is not None and dim.value is not None:
                size *= int(dim)
        if size > 500:
            print(tensor.name, tensor.shape, size)


class Config:
    def __init__(self):
        self.num_epochs = 50
        self.batch_size = 64
        self.ckpt_dir = './checkpoints/cspn'
        self.model_name = 'cspn'


class CspnTrainer:
    def __init__(self, spn, data, x_ph, train_ph, conf, sess=tf.Session()):
        self.spn, self.data, self.x_ph = spn, data, x_ph
        self.conf, self.sess = conf, sess

        self.y_ph = tf.placeholder(tf.float32,
                                   [conf.batch_size] + list(data.train_y.shape[1:]),
                                   name="y_ph")
        self.train_ph = train_ph
        spn_input = tf.reshape(self.y_ph, [conf.batch_size, -1])
        self.marginalized = tf.placeholder(tf.float32, spn_input.shape, name="marg_ph")
        self.spn_output = spn.forward(spn_input, self.marginalized)
        self.loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(self.spn_output, axis=1))
        lr = 1e-3
        print("Learning Rate: {}".format(lr))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()
        if os.path.exists(conf.ckpt_dir):
            self.saver.restore(self.sess, conf.ckpt_dir + os.path.basename(os.path.normpath(conf.ckpt_dir)))
            print('\n **** \n Loaded parameters \n **** \n')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialized parameters')

        i = 0
        log_path = 'results/run0'
        while os.path.exists(log_path):
            log_path = 'results/run{}'.format(i)
            i += 1
        os.makedirs(log_path)
        dump_attributes(conf, log_path + '/conf.csv')
        self.log_path = log_path
        self.log_file = open(log_path + '/results.csv', 'a')

    def run_training(self, no_save=False, each_iter=False):
        batch_size = self.conf.batch_size
        batches_per_epoch = self.data.train_y.shape[0] // batch_size
        loss_curve = []
        for i in range(self.conf.num_epochs):
            for j in range(batches_per_epoch):
                x_batch = self.data.train_x[j * batch_size : (j + 1) * batch_size, :]
                y_batch = self.data.train_y[j * batch_size : (j + 1) * batch_size, :]
                feed_dict = {self.x_ph: x_batch,
                             self.y_ph: y_batch,
                             self.marginalized: np.zeros(self.marginalized.shape),
                             self.train_ph: True}

                _, cur_output, cur_loss = self.sess.run(
                    [self.train_op, self.spn_output, self.loss], feed_dict=feed_dict)

                if j % 100 == 0:
                    print('ep. {}, batch {}, train ll {:.2f}                  '.format(i, j, -cur_loss), end='\r', flush=True)
                    loss_curve.append(-cur_loss)
                    # TODO: below is just a copy of what is used at the end of example_script.py - maybe combine into one
                    show_immediate_success = False
                    if show_immediate_success:
                        mpe = self.spn.reconstruct_batch(feed_dict, self.sess)
                        if len(y_batch.shape) == 2:
                            dims = y_batch.shape[1]
                        else:
                            dims = 1
                        result = np.hstack((y_batch, mpe))
                        if "Gauss" in str(self.spn.vector_list[0][0]):
                            result = np.round(result)
                        print("Train Batch Hits: {}".format(((result[:, :dims] == result[:, dims:dims+dims]).all(axis=1)).sum()))
                        if dims > 1:
                            for i in range(dims):
                                hits = ((result[:, i:i + 1] == result[:, dims + i:dims + i + 1]).all(axis=1)).sum()
                                acc = 100 * (hits / len(y_batch))
                                print('Dimension {} accuracy is {}% ({}/{})'.format(i, acc, hits, len(y_batch)))
                    #import pdb;pdb.set_trace()
                    # self.validate(i, j, -cur_loss)
                    # mpe = data.merge_data(data.test_x[:batch_size], mpe)[..., 0]
                    # original = data.test_im[:batch_size][..., 0]
                    # vis.images(np.expand_dims(original[:16], 1))
                    # vis.images(np.expand_dims(mpe[:16], 1))
                    # mse = np.sum((mpe * 255 - original * 255)**2) / (32 * 64 * batch_size)
            # print(cur_output[:10])
            # acc = num_correct / (batch_size * batches_per_epoch)
            # print(i, acc, cur_loss)
            if i % 2 == 1 and not no_save or each_iter:
                self.saver.save(self.sess, self.conf.ckpt_dir)
                print('Parameters saved')

        return loss_curve

    def validate(self, epoch, iteration, train_ll):
        batch_size = self.conf.batch_size
        num_batches = self.data.test_x.shape[0] // batch_size
        correct_per_feature = np.zeros((self.data.test_y.shape[1],), dtype=np.int32)
        all_correct = 0
        most_correct = 0
        test_ll = 0.
        for i in range(num_batches):
            x_batch = self.data.test_x[i * batch_size: (i+1) * batch_size]
            y_batch = self.data.test_y[i * batch_size: (i+1) * batch_size]
            feed_dict = {self.x_ph: x_batch,
                         self.y_ph: np.zeros_like(y_batch),
                         self.marginalized: np.ones(self.marginalized.shape),
                         self.train_ph: False}
            mpe = self.spn.reconstruct_batch(feed_dict, self.sess)
            mpe = np.reshape(mpe, y_batch.shape)

            correct = (mpe == y_batch)
            totally_correct = np.all(correct, axis=1)
            mostly_correct = np.sum(correct, axis=1) >= 0.8 * self.data.test_y.shape[1]
            all_correct += totally_correct.sum(axis=0)
            most_correct += mostly_correct.sum(axis=0)
            correct_per_feature += correct.sum(axis=0)

            feed_dict[self.marginalized] = np.zeros(self.marginalized.shape)
            feed_dict[self.y_ph] = y_batch
            cur_loss = self.sess.run(self.loss, feed_dict=feed_dict)
            test_ll += cur_loss
        acc_per_feature = correct_per_feature / (num_batches * batch_size)
        test_ll /= -num_batches
        total_acc = acc_per_feature.mean()
        all_correct_acc = all_correct / (num_batches * batch_size)
        most_correct_acc = most_correct / (num_batches * batch_size)
        print('test accuracy per feature')
        print(acc_per_feature)
        print('average accuracy {:.2%}'.format(total_acc))
        print('getting it all right accuracy {:.2%}'.format(all_correct_acc))
        print('getting most of it right accuracy {:.2%}'.format(most_correct_acc))
        print('avg test ll {:.2f}'.format(test_ll))
        self.log_file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
            epoch, iteration, total_acc, all_correct_acc, most_correct_acc, train_ll, test_ll))
        self.log_file.flush()


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_var(shape, name):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                 dtype=tf.float32)
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)


# class MnistDataset:
    # def __init__(self):
        # (self.train_im, self.train_labels), (self.test_im, self.test_labels) = load_mnist()
        # self.train_x, self.train_y = self.split_data(self.train_im)
        # self.test_x, self.test_y = self.split_data(self.test_im)
# 
    # def split_data(self, data):
        # return data[:, :28 * 28//2], data[:, 28 * 28//2:]

def binarize_minst_labels(label_data):
    """
    even numbers are 0
    odd numbers are 1
    """
    for ind, y in enumerate(label_data):
        if y % 2 == 0:
            label_data[ind, 0] = 0
        else:
            label_data[ind, 0] = 1
    return label_data

def multirize_minst_labels(label_data):
    """
    0-2 are 0
    3-5 are 1
    6-8 are 2
    9 is 3
    """
    for ind, y in enumerate(label_data):
        if y in [0,1,2]:
            label_data[ind, 0] = 0
        elif y in [3,4,5]:
            label_data[ind, 0] = 1
        elif y in [6,7,8]:
            label_data[ind, 0] = 2
        else:
            label_data[ind, 0] = 3
    return label_data

class MnistDataset:
    def __init__(self, special_labels=False, binarize=False, multirize=False, multidim=False):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = load_mnist(fashion=False)
        if special_labels:
            self.train_y = np.load('data/mnist-train-labels.npy')
            self.test_y = np.load('data/mnist-test-labels.npy')
        else:
            self.train_y = np.reshape(self.train_y,  (-1, 1))
            self.test_y = np.reshape(self.test_y,  (-1, 1))

        if binarize and not multirize:
            if multidim:
                print('Adding Binarized MNIST Labels to regular Labels.')
                self.train_y = np.hstack((self.train_y, binarize_minst_labels(self.train_y.copy())))
                self.test_y = np.hstack((self.test_y, binarize_minst_labels(self.test_y.copy())))
            else:
                print('Using Binarized MNIST Labels.')
                self.train_y = binarize_minst_labels(self.train_y)
                self.test_y = binarize_minst_labels(self.test_y)
        if multirize and not binarize:
            print('Using Multirized MNIST Labels.')
            self.train_y = multirize_minst_labels(self.train_y)
            self.test_y = multirize_minst_labels(self.test_y)


class FashionDataset:
    def __init__(self, multilabel=False):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = load_mnist(fashion=True)
        self.train_y = np.load('data/fashion-train-labels.npy')
        self.test_y = np.load('data/fashion-test-labels.npy')


class CelebDataset:
    def __init__(self, size=128):
        path = '/work/data/celeba/'
        self.train_x = np.load(path + 'celeb{}-train.npy'.format(size))
        self.test_x = np.load(path + 'celeb{}-test.npy'.format(size))
        self.train_y = np.load(path + 'celeb-train-labels.npy')
        self.test_y = np.load(path + 'celeb-test-labels.npy')
        print('label range', self.train_y.min(), self.train_y.max())


def mnist_completion():
    batch_size = 100
    x_shape = (batch_size, 28, 14)
    y_shape = (batch_size, 28, 14)
    x_dims = y_dims = 28 * 14

    x_ph = tf.placeholder(tf.float32, [batch_size, x_dims])

    if True:
        sum_weights, leaf_weights = build_nn_mnist(x_ph, y_shape, train_ph3000, 64)
        param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)
    else:
        mlp = ReluMLP(x_dims, [1000, 1000, 32256], ['r', 'r', 'l'])
        mlp_output = mlp.forward(x_ph)
        param_provider = RAT_SPN.UnorderedParamProvider(mlp_output)

    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 8):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 8
    args.num_gauss = 4
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
    print("num_params", spn.num_params())

    dataset = MnistDataset()
    sess = tf.Session()
    train_cspn(spn, dataset, x_ph, batch_size=batch_size, num_epochs=1000, sess=sess)


def fashion_mnist_attr(conf):
    batch_size = conf.batch_size
    if conf.dataset == 'celeb':
        x_shape = (batch_size, 128, 128, 3)
        y_shape = (batch_size, 40)
        x_dims = y_dims = 1
        for dim in x_shape[1:]:
            x_dims *= dim
        for dim in y_shape[1:]:
            y_dims *= dim
    else:
        x_shape = (batch_size, 28, 28, 1)
        y_shape = (batch_size, 16)
        x_dims = 28 * 28
        y_dims = 16

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    if conf.model_name == "mdn":
        k = 10
        output_shape = y_shape[1] * k + k
        params = model.build_nn_mnist_baseline(x_ph, (batch_size, output_shape), train_ph)

        spn = model.MixtureDensityNetwork(params, k, y_shape[1])
        conf.ckpt_dir = './checkpoints/fashion-mdn'
    elif conf.model_name == "meanfield":
        print('mean field')
        params = model.build_nn_mnist_baseline(x_ph, y_shape, train_ph)
        spn = model.MeanField(params)
        conf.ckpt_dir = './checkpoints/fashion-meanfield'
    elif conf.model_name == 'cspn':
        sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, 2600, 32)
        param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

        rg = region_graph.RegionGraph(range(y_dims))
        for _ in range(0, 8):
            rg.random_split(2, 2)

        args = RAT_SPN.SpnArgs()
        args.normalized_sums = True
        args.param_provider = param_provider
        args.num_sums = 8
        args.num_gauss = 4
        args.dist = 'Bernoulli'
        spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
        print("TOTAL", spn.num_params())
    else:
        raise ValueError('Unknown model name ' + str(conf.model_name))

    if conf.dataset == 'mnist':
        dataset = MnistDataset()
    elif conf.dataset == 'fashion':
        dataset = FashionDataset()
    elif conf.dataset == 'celeb':
        dataset = CelebDataset()
        conf.num_epochs = 20
    else:
        raise ValueError('Unknown dataset ' + dataset)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training()


def celeb_attr(model_name=""):
    conf = TrainerConfig()
    batch_size = conf.batch_size
    x_shape = (batch_size, 128, 128, 3)
    y_shape = (batch_size, 40)
    x_dims = y_dims = 1
    for dim in x_shape[1:]:
        x_dims *= dim
    for dim in y_shape[1:]:
        y_dims *= dim

    x_ph = tf.placeholder(tf.float32, x_shape)

    if model_name == "mdn":
        k = 10
        output_shape = y_shape[1] * k + k
        params = model.build_nn_celeb_baseline(x_ph, (batch_size, output_shape))

        spn = model.MixtureDensityNetwork(params, k, y_shape[1])
        conf.ckpt_dir = './checkpoints/baseline'
    elif model_name == "meanfield":
        params = model.build_nn_celeb_baseline(x_ph, y_shape)
        spn = model.MeanField(params)
        conf.ckpt_dir = './checkpoints/baseline'
    else:
        sum_weights, leaf_weights = model.build_nn_celeb(x_ph, y_shape, 2600, 32)
        param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

        rg = region_graph.RegionGraph(range(y_dims))
        for _ in range(0, 8):
            rg.random_split(2, 2)

        args = RAT_SPN.SpnArgs()
        args.normalized_sums = True
        args.param_provider = param_provider
        args.num_sums = 8
        args.num_gauss = 4
        args.dist = 'Bernoulli'
        spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
        print("TOTAL", spn.num_params())

    dataset = CelebDataset()

    sess = tf.Session()
    trainer = CspnTrainer(spn, dataset, x_ph, conf, sess=sess)
    trainer.run_training()


if __name__ == "__main__":
    with tf.device('/GPU:0'):
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset')
        parser.add_argument('model')
        args = parser.parse_args()

        conf = Config()
        conf.model_name = args.model
        conf.dataset = args.dataset
        fashion_mnist_attr(conf)


