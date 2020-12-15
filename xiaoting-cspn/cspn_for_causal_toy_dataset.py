import numpy as np
import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer

class CausalDataset():
    def __init__(self, discrete=True):
        if discrete:
            path_data = '../datasets/causal_health_toy_data_discrete.pkl'
        else:
            path_data = 'data/causal_health_toy_data.pkl'
        import pickle
        with open(path_data, "rb") as f:
            data = pickle.load(f)
        data = np.vstack((data['H'], data['M']))
        # random shuffle
        # np.random.seed(0)
        # random_indices = np.random.permutation(np.arange(len(data[0,:])))
        # data = np.array([data[:,ri] for ri in random_indices]).T
        train = data[:,:800]
        test = data[:,800:]
        # we want to model p(F|A)
        self.train_x = train[0,:][:,np.newaxis] # select A
        self.train_y = train[1,:][:,np.newaxis] # select F
        self.test_x = test[0,:][:,np.newaxis] # select A
        self.test_y = test[1,:][:,np.newaxis] # select F

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'
    conf.num_epochs = 100

    batch_size = conf.batch_size
    x_shape = (batch_size, 1)
    y_shape = (batch_size, 1)
    x_dims = 1
    y_dims = 1

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # generate parameters for spn
    sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, 600, 8)
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

    # build spn graph
    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 4):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 4
    args.num_gauss = 2
    args.dist = 'Gauss'#'Categorical'
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    dataset = CausalDataset()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training(no_save=True)

    x_batch = dataset.test_x[: conf.batch_size]
    y_batch = dataset.test_y[: conf.batch_size]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)

    print(np.hstack((y_batch, mpe)))

    '''
    TODO:
        Upon training, simply collect the probability table i.e. for every X,Y combination
        get the probability.
        E.g. in the Gaussian case we would have |X| Gaussians.
        With Categorical we'd have |X| times |Y| values.
        Then save it for loading to a later point.
    '''