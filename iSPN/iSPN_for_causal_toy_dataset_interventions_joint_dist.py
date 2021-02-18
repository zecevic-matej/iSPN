import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer
import pickle
import itertools
import numpy as np
import pandas as pd

class CausalDataset():
    def __init__(self, transform_to_indices=False, seperate_dims=False):
        """
        train_y/test_y are (num_samples, num_vars) matrix
        """
        path_datasets = [
            '../datasets/data_for_uniform_interventions/causal_health_toy_data_discrete_intervention_None_100k.pkl',
            #'../datasets/data_for_uniform_interventions/causal_health_toy_data_discrete_intervention_do(H)=U(H)_100k.pkl'
                         ]
        ys_all = []
        xs_all = []
        for path_data in path_datasets: # collect per intervention
            with open(path_data, "rb") as f:
                data = pickle.load(f)
            order = {0:'A',1:'F',2:'H',3:'M'}
            data = np.vstack((data['A'],data['F'],data['H'],data['M']))
            if seperate_dims:
                lut = {'A': {},'F': {},'H': {},'M': {}}
                for ind_k, k in enumerate(lut.keys()):
                    for ind, y in enumerate(np.unique(data[ind_k,:])):
                        lut[k].update({y: ind})
            else:
                lut = {}
                only_states_with_support = True
                if only_states_with_support:
                    states_with_support = np.unique(data.T, axis=0)
                    for ind, s in enumerate(states_with_support):
                        lut.update({tuple(s.tolist()): ind})
                else:
                    domains = [np.unique(data[i, :]).tolist() for i in range(data.shape[0])]
                    for ind, s in enumerate(itertools.product(*domains)):
                        lut.update({s: ind})
            ys = data.T
            if transform_to_indices:
                if seperate_dims:
                    for i in range(len(ys)):
                        ys[i, :] = np.array([lut[order[dim]][x] for dim, x in enumerate(ys[i, :])])
                else:
                    for i in range(len(ys)):
                        ys[i, :] = lut[tuple(ys[i,:])]
                    ys = ys[:,0][:,np.newaxis]
                self.labels_lut = lut
            else:
                self.labels_lut = None
            if path_data.split('intervention_')[1].split('_')[0] == 'None':
                # A->F, A->H, F->H, H->M
                g = np.array([[0,1,1,0],
                              [0,0,1,0],
                              [0,0,0,1],
                              [0,0,0,0]])
            elif path_data.split('intervention_')[1].split('_')[0] == 'do(H)=U(H)':
                # A->F, A-/->H, F-/->H, H->M
                g = np.array([[0,1,0,0],
                              [0,0,0,0],
                              [0,0,0,1],
                              [0,0,0,0]])
            g.flatten()
            xs = np.tile(g.flatten()[:,np.newaxis],len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
        np.random.seed(0)
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        test_leftout = 10000
        self.train_x = xs_all[:len(random_sorting)-test_leftout,:]
        self.test_x = xs_all[len(random_sorting)-test_leftout:,:]
        self.train_y = ys_all[:len(random_sorting)-test_leftout,:]
        self.test_y = ys_all[len(random_sorting)-test_leftout:,:]
        #import pdb; pdb.set_trace()

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'
    conf.num_epochs = 20#100
    conf.batch_size = 64

    batch_size = conf.batch_size
    x_shape = (batch_size, 16)
    y_shape = (batch_size, 1)
    x_dims = 16
    y_dims = 1

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # generate parameters for spn
    sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, 600, 400)
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
    num_classes = 340#12096
    args.dist = 'Categorical'
    spn = RAT_SPN.RatSpn(num_classes=num_classes, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    dataset = CausalDataset(transform_to_indices=True, seperate_dims=False)

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

    try:
        df_mpe = pd.DataFrame(np.hstack((x_batch, y_batch, mpe)), columns=['X', 'Y', 'MPE(X)'])
        print(df_mpe)
    except Exception as e:
        print('Check the shapes. {}'.format(e))
    result = np.hstack((y_batch, mpe))
    hits = (result[:,0]==result[:,1]).sum()
    print("Accuracy (exact Hit) is {}% ({}/{})".format(100*(hits/len(result)), hits, len(result)))


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