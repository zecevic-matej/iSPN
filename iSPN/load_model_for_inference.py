import numpy as np
import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer
from main import MnistDataset
with tf.device('/GPU:0'):

    conf = Config()
    conf.model_name = 'cspn'
    conf.dataset = 'mnist'
    conf.num_epochs = 1
    conf.ckpt_dir = './checkpoints/cspn-regular-mnist/'

    batch_size = conf.batch_size
    x_shape = (batch_size, 28, 28, 1)
    y_shape = (batch_size, 1)#16)
    x_dims = 28 * 28
    y_dims = 1#16

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

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
    args.dist = 'Gauss' #'Bernoulli'   # choose distribution
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    dataset = MnistDataset()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)

    # random shuffle test data
    np.random.seed(0)
    random_indices = np.random.permutation(np.arange(len(dataset.test_y[:,0])))
    dataset.test_x = np.array([dataset.test_x[ri,:] for ri in random_indices])
    dataset.test_y = np.array([dataset.test_y[ri,:] for ri in random_indices])

    x_batch = dataset.test_x[: conf.batch_size]
    y_batch = dataset.test_y[: conf.batch_size]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)
    #mpe = np.reshape(mpe, y_batch.shape)
    print(np.hstack((y_batch,mpe)))