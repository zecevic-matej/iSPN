import numpy as np
np.set_printoptions(suppress=True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import RAT_SPN
from relu_mlp import ReluMLP
import region_graph
import model
from main import Config, CspnTrainer
from main import MnistDataset

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'
    conf.dataset = 'mnist'
    conf.num_epochs = 1
    conf.ckpt_dir = './checkpoints/none'#'./checkpoints/cspn-regular-mnist'
    #fashion_mnist_attr(conf)

    batch_size = conf.batch_size
    x_shape = (batch_size, 28, 28, 1)
    y_shape = (batch_size, 1)#16)
    x_dims = 28 * 28
    y_dims = 1#16

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # generate parameters for spn
    sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, 2600, 32)
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

    # build spn graph
    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 4):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 8
    args.num_gauss = 4
    args.dist = 'Categorical'   # choose distribution
    print('Using {} vectors'.format(args.dist))
    num_classes = 4#[10,2]
    spn = RAT_SPN.RatSpn(num_classes=num_classes, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())


    if conf.dataset == 'mnist':
        dataset = MnistDataset(binarize=False, multirize=True, multidim=False)
    else:
        raise ValueError('Unknown dataset ' + conf.dataset)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training(each_iter=True)

    x_batch = dataset.test_x[: conf.batch_size]
    y_batch = dataset.test_y[: conf.batch_size]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)
    #mpe = np.reshape(mpe, y_batch.shape)

    if len(y_batch.shape) == 2:
        dims = y_batch.shape[1]
    else:
        dims = 1
    result = np.hstack((y_batch,mpe))
    if args.dist == 'Gauss':
        result = np.round(result)
    hits = ((result[:, :dims] == result[:, dims:dims+dims]).all(axis=1)).sum()
    acc = 100 * (hits / len(y_batch))
    print('Accuracy is {}% ({}/{})'.format(acc, hits, len(y_batch)))
    if dims > 1:
        for i in range(dims):
            hits = ((result[:, i:i+1] == result[:, dims+i:dims+i+1]).all(axis=1)).sum()
            acc = 100 * (hits / len(y_batch))
            print('Dimension {} accuracy is {}% ({}/{})'.format(i, acc, hits, len(y_batch)))