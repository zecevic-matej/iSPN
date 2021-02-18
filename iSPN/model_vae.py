import tensorflow as tf
import numpy as np


def build_feature_extraction_layers(inp):
    with tf.variable_scope('nn'):
        conv1 = tf.layers.conv2d(inputs=inp,
                                 filters=64,
                                 kernel_size=7,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 3, padding='same')
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 3, 3, padding='same')
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=5,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, 3, 3, padding='same')
        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=128,
                                 kernel_size=3,
                                 padding='same',
                                 activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4, 3, 3, padding='same')
        print(pool4.shape)
        linearized = tf.reshape(pool4, (tf.shape(pool4)[0], 2*2*128))
        return linearized


def build_nn_celeb_baseline(inp, latent_space):
    with tf.variable_scope('nn'):
        linearized = build_feature_extraction_layers(inp)
        fc1 = tf.layers.dense(inputs=linearized,
                              units=latent_space*4,
                              activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1,
                              units=latent_space*2,
                              activation=None)
    print_num_vars()
    return fc2


def build_nn_mnist(inp, latent_dim):
    # TODO restile so it fits to the rest
    net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            # tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            # tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim * 2),
        ])
    return net(inp)


def build_nn_celeb_latentspace(inp, output_shape, num_sum_weights, num_leaf_weights, configuration):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    if configuration["hidden1"] > 0:
        inp = tf.layers.dense(inputs=inp,
                              units=configuration["hidden1"],
                              activation=tf.nn.relu)
    with tf.variable_scope("nn_latent"):
        upsampled = build_feature_upsampling_layer_celeb(inp, configuration)

        # TODO Hardcoded SUM Weights filters and reshape
        sum_weights = tf.layers.conv2d(upsampled,
                                       filters=4,
                                       kernel_size=3,
                                       padding='same')

        num_sum_weights = np.prod(sum_weights.get_shape().as_list()[1:])
        sum_weights = tf.reshape(sum_weights, (batch_size, num_sum_weights))

        leaf_weights = tf.layers.conv2d(upsampled,
                                        filters=num_leaf_weights * 3,
                                        kernel_size=3,
                                        padding='same')

        print(leaf_weights)
        print(num_leaf_weights)
        print(output_dims)
        # exit()
        leaf_weights = tf.reshape(leaf_weights, (batch_size, output_dims, num_leaf_weights))

    print(sum_weights)
    print(leaf_weights)
    return sum_weights, leaf_weights


def build_nn_mnist_latentspace(inp, output_shape, num_sum_weights, num_leaf_weights, configuration):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    if configuration["hidden1"] > 0:
        inp = tf.layers.dense(inputs=inp,
                              units=configuration["hidden1"],
                              activation=tf.nn.relu)
    with tf.variable_scope("nn_latent"):
        upsampled = build_feature_upsampling_layer(inp, configuration)

        # TODO Hardcoded SUM Weights filters and reshape
        sum_weights = tf.layers.conv2d(upsampled,
                                       filters=4,
                                       kernel_size=3,
                                       padding='same')

        num_sum_weights = np.prod(sum_weights.get_shape().as_list()[1:])
        sum_weights = tf.reshape(sum_weights, (batch_size, num_sum_weights))

        leaf_weights = tf.layers.conv2d(upsampled,
                                        filters=num_leaf_weights,
                                        kernel_size=3,
                                        padding='same')

        leaf_weights = tf.reshape(leaf_weights, (batch_size, output_dims, num_leaf_weights))

    print(sum_weights)
    print(leaf_weights)
    return sum_weights, leaf_weights


def build_nn_mnist_latentspace_mdn(inp, output_shape, k, configuration):
    batch_size = int(inp.shape[0])
    output_shape = list(output_shape)
    output_dims = 1
    for dim in output_shape[1:]:
        output_dims *= int(dim)

    # preincr = tf.layers.dense(inputs=inp,
    #                          units=configuration["hidden1"],
    #                          activation=tf.nn.relu)
    with tf.variable_scope("nn"):
        upsampled = build_feature_upsampling_layer(inp, configuration)
        mdn_weights = tf.layers.conv2d_transpose(upsampled,
                                            filters=k+1,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same')
        mdn_weights = tf.layers.flatten(mdn_weights)
        # leaf_weights = build_latent_to_leaf(upsampled, batch_size, output_dims, num_leaf_weights, configuration)

    print(mdn_weights)
    return mdn_weights


def build_feature_upsampling_layer(inp, config):
    upscale_shape = list((7, 7, 32))
    upscale_size = np.prod(upscale_shape)
    # for batch size
    upscale_shape.insert(0, -1)
    print(upscale_shape)

    dense1 = tf.layers.dense(inputs=inp,
                             units=upscale_size,
                             activation=tf.nn.relu)

    rescaled = tf.reshape(dense1, upscale_shape)
    print(rescaled)

    tconv1 = tf.layers.conv2d_transpose(rescaled,
                                        filters=config["filters1"],
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)
    tconv2 = tf.layers.conv2d_transpose(tconv1,
                                        filters=config["filters2"],
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)
    return tconv2


def build_feature_upsampling_layer_celeb(inp, config):
    upscale_shape = list((8, 8, 32))
    upscale_size = np.prod(upscale_shape)
    # for batch size
    upscale_shape.insert(0, -1)
    print(upscale_shape)

    dense1 = tf.layers.dense(inputs=inp,
                             units=upscale_size,
                             activation=tf.nn.relu)

    rescaled = tf.reshape(dense1, upscale_shape)
    print(rescaled)

    tconv1 = tf.layers.conv2d_transpose(rescaled,
                                        filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)
    tconv2 = tf.layers.conv2d_transpose(tconv1,
                                        filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)
    tconv3 = tf.layers.conv2d_transpose(tconv2,
                                        filters=256,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)
    tconv4 = tf.layers.conv2d_transpose(tconv3,
                                        filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation=tf.nn.relu)

    return tconv4


def print_num_vars():
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nn')
    num_params = 0
    print('--- parameters ---')
    for var in all_vars:
        num_params += var.shape.num_elements()
        print(var)
    print('The neural network has {} parameters in total'.format(num_params))
