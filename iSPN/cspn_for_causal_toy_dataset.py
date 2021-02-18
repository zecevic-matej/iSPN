import numpy as np
import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer

class CausalDataset():
    def __init__(self, X, Y, discrete=True, transform_to_indices=False):
        if discrete:
            path_data = '../datasets/causal_health_toy_data_discrete.pkl'
        else:
            path_data = 'data/causal_health_toy_data.pkl'
        import pickle
        with open(path_data, "rb") as f:
            data = pickle.load(f)
        self.data = data
        data = np.vstack((data[X], data[Y]))
        # random shuffle
        # np.random.seed(0)
        # random_indices = np.random.permutation(np.arange(len(data[0,:])))
        # data = np.array([data[:,ri] for ri in random_indices]).T
        train = data[:,:800]
        test = data[:,800:]
        lut = {}
        for ind, y in enumerate(np.unique(data[1,:])):
            lut.update({y: ind})
        # we want to model p(F|A)
        self.train_x = train[0,:][:,np.newaxis] # select A
        self.train_y = train[1,:][:,np.newaxis] # select F
        self.test_x = test[0,:][:,np.newaxis] # select A
        self.test_y = test[1,:][:,np.newaxis] # select F
        if transform_to_indices:
            for i in range(len(self.train_y)):
                self.train_y[i, :] = np.array([lut[self.train_y[i, :][0]]])
            for i in range(len(self.test_y)):
                self.test_y[i, :] = np.array([lut[self.test_y[i, :][0]]])
            self.labels_lut = lut
        else:
            self.labels_lut = None

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'
    conf.num_epochs = 20#100
    conf.batch_size = 64

    batch_size = conf.batch_size
    x_shape = (batch_size, 1)
    y_shape = (batch_size, 1)
    x_dims = 1
    y_dims = 1

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # generate parameters for spn
    sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, 600, 12)
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
    num_classes = 12
    args.dist = 'Categorical'
    spn = RAT_SPN.RatSpn(num_classes=num_classes, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    X='H' # 'A'
    Y='M' # 'F'
    transform_to_indices = True # False
    dataset = CausalDataset(X,Y,discrete=True, transform_to_indices=transform_to_indices)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    trainer.run_training(no_save=True)

    # test performance on test set, unseen data
    x_batch = dataset.test_x[: conf.batch_size]
    y_batch = dataset.test_y[: conf.batch_size]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)

    import pandas as pd
    df_mpe = pd.DataFrame(np.hstack((x_batch, y_batch, mpe)), columns=['X', 'Y', 'MPE(X)'])
    result = np.hstack((y_batch, mpe))
    hits = (result[:,0]==result[:,1]).sum()
    print("Accuracy (exact Hit) is {}% ({}/{})".format(100*(hits/len(result)), hits, len(result)))
    print(df_mpe)


    # collect all unique states (X=x) indices
    # assumes all of them appear at least once in train set
    # assumes the number of unique states is smaller than batch size
    ind = [np.where(x == dataset.train_x)[0][0] for x in np.unique(dataset.train_x)]
    remainders = []
    for i in range(batch_size - len(ind)):
        candidate = np.random.choice(np.arange(len(dataset.train_x)))
        while candidate in ind:
            candidate = np.random.choice(np.arange(len(dataset.train_x)))
        remainders.append(candidate)
    x_batch = dataset.train_x[ind + remainders]
    y_batch = dataset.train_y[ind + remainders]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess)
    df_mpe = pd.DataFrame(np.hstack((x_batch, y_batch, mpe)), columns=['X', 'Y', 'MPE(X)'])
    result = np.hstack((y_batch, mpe))
    result = result[:len(ind),:]
    hits = (result[:,0]==result[:,1]).sum()
    print("Accuracy (exact Hit) on Unique States is {}% ({}/{})".format(100*(hits/len(result)), hits, len(result)))
    print(df_mpe)

    # now, create the CPT
    from scipy.special import softmax
    states_X = np.unique(x_batch)
    states_Y = np.unique(y_batch)
    if isinstance(dataset.labels_lut, dict):
        for i in states_Y:
            for k, v in dataset.labels_lut.items():
                if v == i:
                    states_Y[int(i)] = k
    assert(len(states_X) == len(np.unique(dataset.data[X])))
    assert(len(states_Y) == len(np.unique(dataset.data[Y])))

    # collect all unique state distributions
    learned_dists = trainer.spn.output_vector.np_params[:len(states_X)]

    # collect the CPT
    np_pt = np.zeros((len(states_Y), len(states_X)))
    for c, x in enumerate(states_X):
        dist_logits = learned_dists[c, :]
        dist_probs = softmax(dist_logits)[0]
        for r, y in enumerate(states_Y):
            np_pt[r, c] = dist_probs[r]
    pt = pd.DataFrame(np_pt,
                      columns=[X + '={} [{}]'.format(x, ind) for ind, x in enumerate(states_X)],
                      index=[Y + '={} [{}]'.format(y, ind) for ind, y in enumerate(states_Y)])
    save=False
    if save:
        pt.to_csv('../models/probability_table_cspn_causal_toy_dataset_p({}_given_{})_CategoricalCSPN.csv'.format(Y,X))
        print('Saved Probability Table.')

    # visualization of distribution
    import matplotlib.pyplot as plt
    for scale in [False, True]:
        title = 'p({}|{}) with Categorical CSPN'.format(Y,X)
        if scale:
            plt.imshow(np_pt, vmin=0, vmax=1)
            title += '\n(Probability Scale 0-1)'
        else:
            plt.imshow(np_pt)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('{}'.format(X))
        plt.ylabel('{}'.format(Y))
        plt.xticks(np.arange(len(states_X)), states_X)
        plt.yticks(np.arange(len(states_Y)), states_Y)
        plt.show()