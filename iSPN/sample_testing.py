# import stuff
import RAT_SPN
import region_graph
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


def create_simple_spn():
    num_copies = 2
    # create simple spn
    rg = region_graph.RegionGraph(range(4*4))
    for _ in range(0, num_copies):
        rg.random_split(2, 2)

    spn_args = RAT_SPN.SpnArgs()
    spn_args.normalized_sums = True
    # spn_args.param_provider = param_provider
    spn_args.num_sums = 20   # why though?
    spn_args.num_gauss = 3      # int(num_leaf_param / num_copies)

    spn_args.dist = 'Bernoulli'
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=spn_args)
    print("created SPN")
    return spn


def create_mnist_spn():
    num_copies = 8
    # create simple spn
    rg = region_graph.RegionGraph(range(np.prod(x_shape)))
    for _ in range(0, num_copies):
        rg.random_split(2, 3)

    spn_args = RAT_SPN.SpnArgs()
    spn_args.normalized_sums = True
    # spn_args.param_provider = param_provider
    spn_args.num_sums = 20   # why though?
    spn_args.num_gauss = 20      # int(num_leaf_param / num_copies)

    spn_args.dist = 'Bernoulli'
    spn = RAT_SPN.RatSpn(num_classes, region_graph=rg, name="spn", args=spn_args)
    print("created SPN")
    return spn


def load_binarized_mnist(class_only=None):
    """
    Load the MNIST dataset
    Each MNIST image is originally a vector of 784 integers, each of which is
    between 0-255 and represents the intensity of a pixel.
    We model each pixel with a Bernoulli distribution in our model, and we statically
    binarize the dataset.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    x_train /= 255.
    x_test /= 255.

    # Binarization
    x_train[x_train >= .5] = 1.
    x_train[x_train < .5] = 0.
    x_test[x_test >= .5] = 1.
    x_test[x_test < .5] = 0.

    if class_only is not None:
        good_idx = []
        for i in range(y_train.shape[0]):
            if y_train[i] == class_only:
                good_idx.append(i)
        good_idx = np.array(good_idx)
        # Selelct x_train with only good idx
        x_train = x_train[good_idx]
        
    return x_train, y_train, x_test, y_test


def generate_and_save_images(sess, model, epoch):
    print("Generating Image")
    predictions = model.sample(16)
    predictions = tf.reshape(predictions, [-1] + x_shape[0:2] + [num_classes])
    predictions = sess.run(predictions)
    fig = plt.figure(figsize=(4, 4))

    # print(predictions)
    # predictions = np.zeros([16, 4, 4, 1])
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='binary', vmin=0, vmax=1)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./gen_images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


# look at architecture and understand what is going on
def tf_test():
    import tf.contrib.distributions as dists

    sess = tf.Session()

    # shape (num_gaus ^ 2, whole scope)
    # num gaus^2 = number of classes (weigths first dimension)

    inputs = np.array([[[0,0,1,0,1,0],
                       [0,0,0,1,0,0],
                       [0,0,1,1,1,0],
                       [0,0,1,1,1,0]]])
    # dist = dists.Categorical(probs=[0.1, 0.4, 0.9, 5])
    # shape (4,3)
    # (Number of different distr, number of classes)
    dist = dists.Categorical(logits=np.log([[0.5, 0.1, 0.4],
                                           [0.1, 0.9, 0.1],
                                           [0.5, 0.1, 0.4],
                                           [0.1, 0.9, 0.1]]))

    # Num weights is arbitrarely chosen...
    print(np.array([[0.1,0.1,0.1],[0.3,0.3,3]]).shape)
    print(dist)
    sampled_indices = np.array([[1,2,3,1],
                                [0,2,1,1],
                                [2,2,3,0]])

    sampled_indices = np.array([[1, 2, 3, 1]])

    num_samples = 1
    num_sums = 4

    # indices = dist.sample(num_samples)
    indices = sampled_indices

    # shape: 4,2 -> (num_samples, num_different_distr), value is the chosen class index

    case_idx = tf.tile(tf.expand_dims(tf.range(num_samples), -1), [1, num_sums])
    # shape: 3,4 -> num_samples, num_different_distr
    # 0,0 is from the first sample and the first distr
    print("case idx:", case_idx)

    # full_idx = tf.stack((case_idx, sampled_indices), axis=-1)
    full_idx = tf.stack((case_idx, indices), axis=-1)

    # each element defines a slice of the input
    result = tf.gather_nd(inputs, full_idx)


    # print(indices)
    sess.run(tf.global_variables_initializer())
    # print(sess.run(indices))
    # print("sampled idx:\n", sampled_indices)
    print("case idx:\n", sess.run(case_idx))
    print("fulludx:\n", sess.run(full_idx))
    # print("indices:\n", sess.run(indices), indices)
    print("result\n", sess.run(result), result)
    exit()


def train_spn(spn, sess):
    batch_size = 128
    input_shape = x_shape
    num_epochs = 50

    # x = np.ones([100] + input_shape)
    # mnist_fake = np.ones([1000] + input_shape)
    train_x, train_y, _, _ = load_binarized_mnist(class_only)
    print(train_x.shape)
    print(train_y.shape)
    x = train_x.astype("float32")
    y = train_y.astype("int32")
    # x = mnist_fake
    # TODO mnist binarization

    x_ph = tf.placeholder(tf.float32,
                          [batch_size] + input_shape,
                          name="x_ph")
    y_ph = tf.placeholder(tf.int32,
                          [batch_size],
                          name="y_ph")

    spn_input = tf.reshape(x_ph, [batch_size, -1])

    marginalized = tf.placeholder(tf.float32, spn_input.shape, name="marg_ph")
    spn_output = spn.forward(spn_input, marginalized)
    # spn_output = spn.forward(spn_input)

    if True:
        # Generative loss for each individual class

        # match correct label idx with number of batch
        label_idx = tf.stack([tf.range(batch_size), y_ph], axis=1)

        # gather_nd to select only from the batches tho correct labels
        loss = tf.reduce_mean(-1 * tf.gather_nd(spn_output, label_idx))
    else:
        loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(spn_output, axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    # TODO: Loss related to class

    # TF STUFF
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # TRAIN
    batch_size = batch_size
    batches_per_epoch = x.shape[0] // batch_size

    if args.debug:
        batches_per_epoch = 2
        num_epochs = 9

    print("Start Training")
    for i in range(num_epochs):
        for j in range(batches_per_epoch):
            x_batch = x[j * batch_size: (j + 1) * batch_size, :]
            y_batch = y[j * batch_size: (j + 1) * batch_size]
            feed_dict = {x_ph: x_batch,
                         y_ph: y_batch,
                         marginalized: np.zeros(marginalized.shape)}

            _, cur_output, cur_loss = sess.run(
                [train_op, spn_output, loss], feed_dict=feed_dict)

        # generate_and_save_images(sess, spn, i)
        saver.save(sess, './my-model')
        print(f"{i:05d}|: Loss: ", np.squeeze(cur_loss))
        # TODO sample for validation


def sample_from_save(spn, sess):

    saver = tf.train.Saver()
    saver.restore(sess, "./my-model")
    for i in range(10):
        generate_and_save_images(sess, spn, 777 + i)
    exit()

# sample ...
def main():
    # tf_test()

    # spn = create_simple_spn()
    spn = create_mnist_spn()

    # print(spn)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #    generate_and_save_images(sess, spn, -1)

    # sample = spn.sample(10, seed=42414345)
    # print("Sampled without training:")
    # sampled_val = np.squeeze(sess.run(sample))
    # print(sampled_val)

    # TODO it seems to run on 1s okay ish, although it still struggels
    # to predict only ones, try out with MNIST sized ones and
    # then on real MNIST images
    # TODO sample once per epoch
    # TODO learn on samples?
    # Da samples so einfach, ist höhere learning rate viel schneller

    # ------- TRAIN SPN --------
    sess = tf.Session()
    # sample_from_save(spn, sess)

    train_spn(spn, sess)
    # sample = spn.sample(10, seed=42414345)
    print("Sampled with training:")
    # sampled_val = np.squeeze(sess.run(sample))
    # print(sampled_val)

    # look at architecture


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser(description="Conditional SPN over latent space z")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Runs training on mininmal settings for debugging")

    args = parser.parse_args()

    class_only = 1
    num_classes = 10
    x_shape = [28, 28, 1]
    lr = 2e-3

    print(f"\n-------Settings\n{args}\n--------\n")
    main()
