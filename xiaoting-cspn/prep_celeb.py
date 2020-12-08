import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os

import pickle

size = 128
dir_data      = "./work/data/img_align_celeba/"
Ntrain        = 180000
Ntest         = 20000
nm_imgs       = np.sort(os.listdir(dir_data))
## name of the jpg files for training set
nm_imgs_train = nm_imgs[:Ntrain]
## name of the jpg files for the testing data
nm_imgs_test  = nm_imgs[Ntrain:Ntrain + Ntest]
img_shape     = (size, size, 3)

def get_npdata(nm_imgs_train):
    X_train = []
    for i, myid in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + myid,
                                          target_size=img_shape[:2])
        image = img_to_array(image)
        X_train.append(image)
        if i % 1000 == 0:
            print(i, '...')
    X_train = np.array(X_train)
    return(X_train)

def load_labels(path):
    f = open(path, 'r')
    n = int(f.readline())
    labels = np.zeros((n, 40), dtype=np.int32)
    f.readline() # skip column headers
    for i in range(n):
        row = f.readline().split()
        # Map labels from {-1, 1} to {0, 1} and skip the first column, which contains the filename
        row = [(int(x) + 1) // 2 for x in row[1:]]
        labels[i] = row
        if i % 10000 == 0:
            print(i, row)
    ratios = np.sum(labels, axis=0) / labels.shape[0]
    print('ratio of ones for each label')
    print(ratios)
    bayes_acc = np.maximum(ratios, 1-ratios)
    print('Bayes acc', bayes_acc.mean())

    return labels


labels = load_labels('list_attr_celeba.txt')
Y_train = labels[:Ntrain]
np.save('celeb-train-labels', Y_train)
Y_test = labels[Ntrain:]
np.save('celeb-test-labels', Y_test)
print('labels saved')

X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))
np.save('celeb{}-train'.format(size), X_train)

X_test  = get_npdata(nm_imgs_test)
print("X_test.shape = {}".format(X_test.shape))
np.save('celeb{}-test'.format(size), X_test)
