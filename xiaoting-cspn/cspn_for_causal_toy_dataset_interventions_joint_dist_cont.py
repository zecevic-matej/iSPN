import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer
import pickle
import os
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
np.set_printoptions(suppress=True)

def get_graph_for_intervention_CHD(intervention):
    """
    for the causal health dataset. get graph according to intervention.
    """
    if intervention == 'None':
        # A->F, A->H, F->H, H->M
        g = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
    elif intervention == 'do(H)=U(H)':
        # A->F, A-/->H, F-/->H, H->M
        g = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
    elif 'do(F)' in intervention:
        # A-/->F, A->H, F->H, H->M
        g = np.array([[0, 0, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
    elif "do(A)" in intervention:
        # A->F, A->H, F->H, H->M
        g = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
    elif "do(M)" in intervention:
        # A->F, A->H, F->H, H-/->M
        g = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    g = g.flatten()
    return g

class CausalHealthDataset():
    def __init__(self, batch_size):
        """
        train_y/test_y are (num_samples, num_vars) matrix
        """
        path_datasets = ("../datasets/data_for_uniform_interventions_continuous/",
            [
                #'causal_health_toy_data_continuous_intervention_do(F)=N(-5,0.1)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(F)=N(-5,0.1)_N1000.pkl',
                'causal_health_toy_data_continuous_intervention_None_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(A)=U(A)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(F)=U(F)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(H)=U(H)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(M)=U(M)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(F)=U(F)_N1000.pkl',
                         ])
        path_datasets = [os.path.join(path_datasets[0], p) for p in path_datasets[1]]
        self.intervention = []
        ys_all = []
        xs_all = []
        for path_data in path_datasets: # collect per intervention
            with open(path_data, "rb") as f:
                data = pickle.load(f)
            self.data = data
            ys = np.vstack((data['A'],data['F'],data['H'],data['M'])).T
            intervention = path_data.split('intervention_')[1].split('_')[0]
            g = get_graph_for_intervention_CHD(intervention)
            self.intervention.append(intervention)
            xs = np.tile(g.flatten()[:,np.newaxis],len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
        np.random.seed(0)
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        test_leftout = int(0.15 * len(random_sorting))
        assert(test_leftout <= len(random_sorting) * 0.2 and test_leftout > batch_size)
        self.train_x = xs_all[:len(random_sorting)-test_leftout,:]
        self.test_x = xs_all[len(random_sorting)-test_leftout:,:]
        self.train_y = ys_all[:len(random_sorting)-test_leftout,:]
        self.test_y = ys_all[len(random_sorting)-test_leftout:,:]
        self.description = 'Causal Health (Cont.)'
        #import pdb; pdb.set_trace()

class BnLearnDataset():
    def __init__(self, bif, batch_size, description):
        p = '../datasets/other/benchmark_data_for_uniform_interventions/{}_uniform_interventions_N10000.pkl'.format(bif)
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.data = data
        self.intervention = []
        ys_all = []
        xs_all = []
        for ind, dict_interv in enumerate(self.data):
            interv = dict_interv['interv']
            self.intervention.append(interv)
            ys = np.array(dict_interv['data'])
            g = np.array(dict_interv['adjmat'],dtype=float).flatten()
            xs = np.tile(g.flatten()[:, np.newaxis], len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
        np.random.seed(0)
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        test_leftout = int(0.15 * len(random_sorting))
        assert(test_leftout <= len(random_sorting) * 0.2 and test_leftout > batch_size)
        self.train_x = xs_all[:len(random_sorting)-test_leftout,:]
        self.test_x = xs_all[len(random_sorting)-test_leftout:,:]
        self.train_y = ys_all[:len(random_sorting)-test_leftout,:]
        self.test_y = ys_all[len(random_sorting)-test_leftout:,:]
        self.description = description

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'

    #################################################################

    # parameters to be adapted
    conf.num_epochs = 130#20 # Causal Health: 130
    conf.batch_size = 1000#100 # Causal Health: 1000
    num_sum_weights = 600#2400 # Causal Health: 600
    num_leaf_weights = 12#96 # Causal Health: 12
    use_simple_mlp = True # Causal Health: True
    bnl_dataset = None #'asia' # Causal Health: None
    dataset_name = bnl_dataset if bnl_dataset else 'CH' # adapt for other datasets
    description = 'Causal Health (Cont.)'#'{} (bnlearn, Bernoulli)'.format(bnl_dataset) # Causal Health: 'Causal Health (Cont.)'
    dataset = CausalHealthDataset(conf.batch_size)#BnLearnDataset(bnl_dataset, conf.batch_size, description) # CausalHealthDataset(batch_size)

    # low, high are the sample range for getting the Probability Density Function (pdf)
    low=-20#-0.5 # Causal Health: -20
    high=100#1.5 # Causal Health: 100
    save_dir="iSPN_trained_uniform_interventions_{}/".format(dataset_name) # if not specified, then plots are plotted instead of saved

    plot_mpe = False
    plot_convex_hull = True

    #################################################################

    x_dims = dataset.train_x.shape[1]
    y_dims = dataset.train_y.shape[1]
    batch_size = conf.batch_size
    x_shape = (batch_size, x_dims)
    y_shape = (batch_size, y_dims)

    x_ph = tf.placeholder(tf.float32, x_shape)
    train_ph = tf.placeholder(tf.bool)

    # generate parameters for spn
    # bnl_datasets and the causal health data use the simple (non-conv) mlp
    sum_weights, leaf_weights = model.build_nn_mnist(x_ph, y_shape, train_ph, num_sum_weights, num_leaf_weights, use_simple_mlp=use_simple_mlp)
    param_provider = RAT_SPN.ScopeBasedParamProvider(sum_weights, leaf_weights)

    # build spn graph
    rg = region_graph.RegionGraph(range(y_dims))
    for _ in range(0, 4):
        rg.random_split(2, 2)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.param_provider = param_provider
    args.num_sums = 4
    args.num_gauss = 4
    args.dist = 'Gauss'
    spn = RAT_SPN.RatSpn(num_classes=1, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    loss_curve = trainer.run_training(no_save=True)

    # test performance on test set, unseen data
    x_batch = dataset.test_x[: conf.batch_size]
    y_batch = dataset.test_y[: conf.batch_size]
    feed_dict = {trainer.x_ph: x_batch,
                 trainer.y_ph: np.zeros_like(y_batch),
                 trainer.marginalized: np.ones(trainer.marginalized.shape),
                 trainer.train_ph: False}
    sample = False
    mpe = trainer.spn.reconstruct_batch(feed_dict, trainer.sess, sample=sample)
    print('****************\nMost Probable Explanation: (with Sample={})\n{}'.format(sample,mpe))

    plt.plot(range(len(loss_curve)), loss_curve)
    plt.title('Intervention: {}\nTraining Gaussian CSPN on {} with {} Samples'.format(dataset.intervention, dataset.description, len(dataset.train_x)))
    plt.ylabel('Log-Likelihood')
    plt.xlabel('Batch #')
    plt.show()


    def compute_pdf(dim, N, low, high, intervention, visualize=False, dd=None, comment=None):
        if bnl_dataset:
            dict_interv = next(item for item in dataset.data if item["interv"] == intervention)
            g = np.array(dict_interv['adjmat'],dtype=float).flatten()
        else:
            g = get_graph_for_intervention_CHD(intervention)
        y_batch = np.zeros((N,y_dims))#4))
        y_batch[:,dim] = np.linspace(low,high,N)
        marginalized = np.ones((N,y_dims))#4))
        marginalized[:,dim] = 0. # seemingly one sets the desired dim to 0
        x_batch = np.tile(g,(N,1))#dataset.train_x[:N]
        feed_dict = {trainer.x_ph: x_batch,
                     trainer.y_ph: y_batch,
                     trainer.marginalized: marginalized,
                     trainer.train_ph: False} # this is important, also used for MPE, makes things deterministic
        out = sess.run(trainer.spn_output, feed_dict=feed_dict)
        #print("+++++++++ SPN OUTPUT MARGINALIZED\n{}".format(out))
        if visualize:
            fig = plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.plot(y_batch[:, dim], out)
            plt.title("Log Likelihood")
            plt.subplot(1,2,2)
            plt.plot(y_batch[:, dim], np.exp(out))
            plt.title("Likelihood")
            if dd:
                name = dd[dim]
            else:
                name = "Variable " + str(dim)
            plt.suptitle("For {} ({})\n{}".format(name, name[0], comment))
            plt.show()
        vals_x = y_batch[:,dim] # range (all sampled points along axis)
        vals_pdf = np.exp(out) # given that out is log-likelihood, take exp() and these are the likelihoods for pdf
        return vals_x, vals_pdf

    def compute_single_pdf(x,dim,N):
        # this is the single PDF for a specific variable "dim"
        # just set N==conf.batch_size
        # doing:
        #
        #   import scipy.integrate as integrate
        #   integrate.quad(lambda x: compute_single_pdf(x,1,1000), -10, 10)[0]
        #
        # gives as expected 1, if dim=1 is "F" where do(F=N(-5,0.1))
        """
        Full Example:
            dim = 1 # is F in this case
            low = -20
            high = 20
            N = 1000
            xs = np.linspace(low, high, N)
            import scipy.integrate as integrate
            integral = integrate.quad(lambda x: compute_single_pdf(x,dim,N), low, high)[0]
            plt.plot(xs, [compute_single_pdf(x, dim, N) for x in xs],
                     label="Integrates on ({},{}) to {}".format(low, high, np.round(integral)))
            plt.title("Learned PDF from CSPN")
            plt.legend()
            plt.show()
        """
        y_batch = np.zeros((N,4))
        y_batch[0,dim] = x
        marginalized = np.ones((N,4))
        marginalized[:,dim] = 0. # seemingly one sets the desired dim to 0
        x_batch = dataset.train_x[:N]
        feed_dict = {trainer.x_ph: x_batch,
                     trainer.y_ph: y_batch,
                     trainer.marginalized: marginalized,
                     trainer.train_ph: False} # this is important, also used for MPE, makes things deterministic
        out = sess.run(trainer.spn_output, feed_dict=feed_dict)
        return np.exp(out[0,:][0])

    colors = ['pink', 'blue', 'orange', 'lightgreen', 'yellow', 'red', 'cyan', 'purple'] # ASSUMES: NEVER more than 8 variables
    N = batch_size
    if bnl_dataset:
        list_vars_per_interv = [dataset.data[i]['data'].columns.tolist() for i in range(len(dataset.intervention))]
        fig_len_x = 2
        fig_len_y = 4
    else:
        print('Assuming Causal Health Dataset.')
        fig_len_x = 2
        fig_len_y = 2
    for ind, interv_desc in enumerate(dataset.intervention):
        fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(12, 10))
        if bnl_dataset:
            comment = ''
            data = dataset.data[ind]['data']
            variables = list_vars_per_interv[0] # assumes that all datasets have the same variables, makes sure that ordering is consistent
        else:
            pp = '../datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_{}_N100000.pkl'.format(interv_desc)
            ppN = int(pp.split("_N")[1].split(".pkl")[0]) if interv_desc != "None" else int(pp.split("None")[1].split("_N")[1].split(".pkl")[0])
            comment = "Histograms on {} K samples, Training on {} K samples".format(np.round(ppN/ 1000,decimals=1), np.round(dataset.train_x.shape[0] / 1000,decimals=1))
            with open(pp, "rb") as f:
                data = pickle.load(f)
            variables = ['Age', 'Food Habits', 'Health', 'Mobility']
            list_vars_per_interv = [variables] * len(dataset.intervention)
        for ind_d, d in enumerate(variables):
            # gt distribution
            if bnl_dataset:
                hist_data = data[d]
            else:
                hist_data = data[d[0]]
            weights = np.ones_like(hist_data)/len(hist_data)
            h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d], label="GT", weights=weights, edgecolor=colors[ind_d])
            axs.flatten()[ind_d].set_title('{}'.format(d))
            axs.flatten()[ind_d].set_xlim(low, high)
            # mpe
            if plot_mpe:
                xc = np.round(mpe[0,ind_d], decimals=1)
                axs.flatten()[ind_d].axvline(x=xc, label='CSPN Max = {}'.format(xc), c='red')
            # pdf
            vals_x, vals_pdf = compute_pdf(list_vars_per_interv[ind].index(d), N, low, high, intervention=interv_desc, visualize=False)
            axs.flatten()[ind_d].plot(vals_x, vals_pdf, label="CSPN learned PDF", color="black", linestyle='solid', linewidth=1.5)
            # convex hull
            if plot_convex_hull:
                points = np.vstack((vals_x,vals_pdf[:,0])).T
                hull = ConvexHull(points)
                for ind_s, simplex in enumerate(hull.simplices):
                    label = 'Convex Hull' if ind_s == 0 else None
                    axs.flatten()[ind_d].plot(points[simplex, 0], points[simplex, 1], 'k-.',label=label)
            axs.flatten()[ind_d].legend(prop={'size':9})
        plt.suptitle('Intervention: {}, sampled {} steps over({},{}), hist_n_bins={}, Train LL {:.2f}\n{}'.format(interv_desc,N,low,high, len(h[0]), np.round(loss_curve[-1],decimals=2), comment))
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_loc = os.path.join(save_dir, "iSPN_int_{}.png".format(interv_desc))
            plt.savefig(save_loc)
            print('Saved @ {}'.format(save_loc))
        else:
            plt.show()