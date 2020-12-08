import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

import main
import RAT_SPN
import region_graph

batch_size = 50


class FacesDataset:
    def __init__(self, mask):
        self.mask = mask
        faces = fetch_olivetti_faces()
        imgs = np.expand_dims(faces['images'], -1)
        print("faces loaded", imgs.shape)
        # self.train_im, self.test_im = train_test_split(imgs, test_size=0.2)
        self.train_im, self.test_im = imgs[:300], imgs[300:]
        np.random.shuffle(self.train_im)
        np.random.shuffle(self.test_im)

        self.train_x, self.train_y = self.split_data(self.train_im)
        self.test_x, self.test_y = self.split_data(self.test_im)

    def split_data(self, data):
        if self.mask == 'left':
            return data[:, :, 32:], data[:, :, :32]
        elif self.mask == 'bottom':
            return data[:, :32, :], data[:, 32:, :]
        else:
            raise ValueError('Mask must be left or bottom')

    def merge_data(self, x, y):
        if self.mask == 'left':
            return np.concatenate((y, x), axis=2)
        elif self.mask == 'bottom':
            return np.concatenate((x, y), axis=1)


def faces_completion(mask='left'):
    if mask == 'left':
        x_shape = (batch_size, 64, 32)
        y_shape = (batch_size, 64, 32)
    elif mask == 'bottom':
        x_shape = (batch_size, 32, 64)
        y_shape = (batch_size, 32, 64)
    else:
        raise ValueError('Mask must be left or bottom')

    x_dims = y_dims = 64 * 32
    x_ph = tf.placeholder(tf.float32, list(x_shape) + [1])

    sum_weights, leaf_weights = main.build_nn(x_ph, y_shape, 3000, 64)
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

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

    dataset = FacesDataset(mask)

    sess = tf.Session()

    main.train_cspn(spn, dataset, x_ph, batch_size=batch_size, num_epochs=1000, sess=sess)


faces_completion('left')
