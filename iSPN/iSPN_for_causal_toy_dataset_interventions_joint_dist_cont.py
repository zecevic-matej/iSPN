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
import gc
import random

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
    elif 'do(H)' in intervention:
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
    def __init__(self, batch_size, extra):
        """
        train_y/test_y are (num_samples, num_vars) matrix
        """
        common_dir = "../datasets/data_for_uniform_interventions_continuous/"
        path_datasets = (common_dir,
            [
                # #'causal_health_toy_data_continuous_intervention_do(F)=N(-5,0.1)_N100000.pkl',
                # #'causal_health_toy_data_continuous_intervention_do(F)=N(-5,0.1)_N1000.pkl',
                # 'causal_health_toy_data_continuous_intervention_None_N100000.pkl',
                # 'causal_health_toy_data_continuous_intervention_do(A)=U(A)_N100000.pkl',
                # 'causal_health_toy_data_continuous_intervention_do(F)=U(F)_N100000.pkl',
                # 'causal_health_toy_data_continuous_intervention_do(H)=U(H)_N100000.pkl',
                # 'causal_health_toy_data_continuous_intervention_do(M)=U(M)_N100000.pkl',
                # #'causal_health_toy_data_continuous_intervention_do(F)=U(F)_N1000.pkl',

                # for appendix ablation
                'causal_health_toy_data_continuous_intervention_None_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(A)=U(A)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(F)=U(F)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(M)=U(M)_N100000.pkl',

                #'causal_health_toy_data_continuous_intervention_do(H)=U(H)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=N(70,10)_N10000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=SBeta(2,2)_N10000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=Gamma(2,10)_N10000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=[25,75]_N10000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=50_N10000.pkl',
                         ])
        path_datasets = [os.path.join(path_datasets[0], p) for p in path_datasets[1]]
        path_datasets.append(os.path.join(common_dir, extra))
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
    def __init__(self, bif, batch_size, description, whitelist):
        p = '../datasets/other/benchmark_data_for_uniform_interventions/{}_uniform_interventions_N100000.pkl'.format(bif)
        with open(p, "rb") as f:
            data = pickle.load(f)
        lut_interventions = {}
        for ind, d in enumerate(data):
            lut_interventions.update({d["interv"]: ind})
        if whitelist is None:
            whitelist = list(lut_interventions.keys())
        indices = [lut_interventions[interv] for interv in whitelist]
        data = [data[i] for i in indices]
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


    for extra in [
                #'causal_health_toy_data_continuous_intervention_do(H)=U(H)_N100000.pkl',
                'causal_health_toy_data_continuous_intervention_do(H)=50_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=N(70,10)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=SBeta(2,2)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=Gamma(2,10)_N100000.pkl',
                #'causal_health_toy_data_continuous_intervention_do(H)=[25,75]_N100000.pkl',
                                  ]:
        extra_dsc = extra.split('do(H)=')[1].split("_N1")[0]

        #for epochs in [5, 20, 50, 100, 200]:

        #################################################################

        # parameters to be adapted
        # conf.num_epochs = 5#70 #130 # Causal Health: 130
        # conf.batch_size = 1000 # Causal Health: 1000
        # num_sum_weights = 600 # Causal Health: 600
        # num_leaf_weights = 12 # Causal Health: 12
        # use_simple_mlp = True # Causal Health: True
        # bnl_dataset = None # Causal Health: None
        # whitelist = None #[None, "lung"]  # only BNL datasetes: all interventions to be considered, if None then everything is considered
        # dataset_name = bnl_dataset if bnl_dataset else f'CH_{extra_dsc}' # adapt for other datasets
        # description = 'Causal Health (Cont.)' # Causal Health: 'Causal Health (Cont.)'
        # dataset = CausalHealthDataset(conf.batch_size, extra=extra) # CausalHealthDataset(conf.batch_size)
        # # low, high are the sample range for getting the Probability Density Function (pdf)
        # low=-20 # Causal Health: -20
        # high=100 # Causal Health: 100

        conf.num_epochs = 50
        conf.batch_size = 100
        num_sum_weights = 500#600#2400
        num_leaf_weights = 24#32#96
        use_simple_mlp = True
        bnl_dataset = 'asia'
        whitelist = None
        dataset_name = bnl_dataset
        description = '{} (bnlearn, Bernoulli)'.format(bnl_dataset)
        dataset = BnLearnDataset(bnl_dataset, conf.batch_size, description, whitelist)
        suffix = "_100k"
        low=-0.5
        high=1.5

        save_dir=f"runtimes/timed_iSPN_trained_uniform_interventions_{dataset_name}_ep{conf.num_epochs}{suffix}/" # if not specified, then plots are plotted instead of saved

        plot_loss_curve = False
        plot_mpe = False
        plot_convex_hull = False
        show_axes_and_legend = True#False

        without_training_from_file = None #'appendix_ablation/iSPN_trained_uniform_interventions_CH/iSPN_pdfs_per_seed.pkl'

        #################################################################

        baseline = None #'../results/baseline_CBN.pkl' # Causal Baseline, used for Motivation Figure

        master_seeds = [606, 1011, 3004]#, 5555, 12096]

        batch_size = conf.batch_size
        if without_training_from_file is None:
            vals_per_seed = {}
            loss_curves_per_seed = []
            for cur_seed in master_seeds:

                np.random.seed(cur_seed)
                tf.set_random_seed(cur_seed)
                tf.random.set_random_seed(cur_seed)
                random.seed(cur_seed)

                x_dims = dataset.train_x.shape[1]
                y_dims = dataset.train_y.shape[1]
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
                args.num_sums = 2#4
                args.num_gauss = 2#4
                args.dist = 'Gauss'
                spn = RAT_SPN.RatSpn(num_classes=1, region_graph=rg, name="spn", args=args)
                print("TOTAL", spn.num_params())

                import time
                t0 = time.time()
                sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=False))
                trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
                loss_curve = trainer.run_training(no_save=True)
                training_time = time.time() - t0
                print(f'TIME {training_time:.2f}')

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

                if plot_loss_curve:
                    plt.plot(range(len(loss_curve)), loss_curve)
                    plt.title('Intervention: {}\nTraining Gaussian CSPN on {} with {} Samples'.format(dataset.intervention, dataset.description, len(dataset.train_x)))
                    plt.ylabel('Log-Likelihood')
                    plt.xlabel('Epoch #')
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

                # collect all data beforehand
                N = batch_size
                if bnl_dataset:
                    list_vars_per_interv = [dataset.data[i]['data'].columns.tolist() for i in range(len(dataset.intervention))]
                    variables = list_vars_per_interv[0]
                else:
                    variables = ['Age', 'Food Habits', 'Health', 'Mobility']
                    list_vars_per_interv = [variables] * len(dataset.intervention)

                vals_per_intervention = {}
                for ind, interv_desc in enumerate(dataset.intervention):
                    vals_per_var = {}
                    for ind_d, d in enumerate(variables):
                        vals_x, vals_pdf = compute_pdf(list_vars_per_interv[ind].index(d), N, low, high, intervention=interv_desc,visualize=False)
                        vals_per_var.update({d: (vals_x, vals_pdf)})
                    vals_per_intervention.update({interv_desc: vals_per_var})
                vals_per_seed.update({cur_seed: (vals_per_intervention, training_time) })

                loss_curves_per_seed.append(loss_curve)

                # clear the session, to re-run
                tf.reset_default_graph()

                if save_dir:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_loc = os.path.join(save_dir, "runtimes.txt")
                    if os.path.exists(save_loc):
                        m = "a"
                    else:
                        m = "w"
                    with open(save_loc, m) as f:
                        f.write(f"{training_time:.2f}\n")


            vals_per_seed.update({"loss curve": (np.mean(loss_curves_per_seed, axis=0), np.std(loss_curves_per_seed, axis=0))})
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_loc = os.path.join(save_dir, "iSPN_pdfs_per_seed.pkl")
                with open(save_loc, "wb") as f:
                    pickle.dump(vals_per_seed, f)

        if without_training_from_file is not None and os.path.exists(without_training_from_file):
            with open(without_training_from_file, "rb") as f:
                vals_per_seed = pickle.load(f)
                print('LOADED SUCCESSFULLY FROM FILE.')
            vals_x = vals_per_seed[606]['None']['Age'][0] # TODO: make this more general

        mean_loss_curve = vals_per_seed['loss curve'][0]
        std_loss_curve = vals_per_seed['loss curve'][1]
        plt.plot(range(1,len(mean_loss_curve)+1), mean_loss_curve, color='#267a85', linestyle='solid', linewidth=1.5, label=f'{extra_dsc}')
        plt.fill_between(range(1,len(mean_loss_curve)+1), mean_loss_curve - std_loss_curve, mean_loss_curve + std_loss_curve, alpha=0.5, edgecolor='#267a85', facecolor='#37b4c7')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "iSPN_mean_loss_curve.png"))
        plt.clf()
        plt.close()

        # vals_x assumed to be same everywhere and not collected here
        #vals_x = vals_per_seed[master_seeds[0]][dataset.intervention[0]][...]
        mean_vals_per_interv = {}
        for ind, interv_desc in enumerate(dataset.intervention):
            mean_vals_per_var = {}
            for v in vals_per_seed[master_seeds[0]][0][interv_desc].keys():
                vals_pdfs = []
                for seed in master_seeds:
                    vals_pdf = vals_per_seed[seed][0][interv_desc][v][1][:,0]
                    vals_pdfs.append(vals_pdf)
                vals_pdfs = np.stack(vals_pdfs)
                mean_vals_per_var.update({v: (np.mean(vals_pdfs, axis=0), np.std(vals_pdfs, axis=0))})
            mean_vals_per_interv.update({interv_desc: mean_vals_per_var})

        # load causalnex baseline
        try:
            if baseline is None or baseline == "":
                raise Exception
            with open(baseline,'rb') as f:
                baseline = pickle.load(f)
        except Exception as e:
            print("No Baseline available or should be used.")
            baseline = None


        #colors = ['pink', 'blue', 'orange', 'lightgreen', 'yellow', 'red', 'cyan', 'purple'] # ASSUMES: NEVER more than 8 variables
        N = batch_size
        if bnl_dataset:
            list_vars_per_interv = [dataset.data[i]['data'].columns.tolist() for i in range(len(dataset.intervention))]
            fig_len_x = 2
            fig_len_y = 4
            color_pairs = [
                ('#e480b5', '#db569d'),
                ('#40b3ef', '#13a0e9'),
                ('#ffbf2e', '#faad00'),
                ('#4fe547', '#26cb1d'),
                ('#ffff6b', '#ffff27'),
                ('#d54b37', '#892b1d'),
                ('#20e6d0', '#15beab'),
                ('#a575dc', '#8b4cd1'),
            ]
            share_x = True
            share_y = True
        else:
            print('Assuming Causal Health Dataset.')
            fig_len_x = 2
            fig_len_y = 2
            color_pairs = [
                ('#ffcfe2', '#ffbed7'),
                ('#ffff6b', '#d1d100'),
                ('#88ed83', '#4fe547'),
                ('#7fccf4','#40b3ef'),
            ]
            share_x = True
            share_y = False
        for ind, interv_desc in enumerate(dataset.intervention):
            fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(12, 10), sharey=share_y, sharex=share_x)
            if bnl_dataset:
                comment = ''
                data = dataset.data[ind]['data']
                variables = list_vars_per_interv[0] # assumes that all datasets have the same variables, makes sure that ordering is consistent
            else:
                # TODO: make more general
                ppN = 100000
                pp = '../datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_{}_N{}.pkl'.format(interv_desc, ppN)
                #ppN = int(pp.split("_N")[1].split(".pkl")[0]) if interv_desc != "None" else int(pp.split("None")[1].split("_N")[1].split(".pkl")[0])
                comment = "Histograms on {} K samples, Training on {} K samples".format(np.round(ppN/ 1000,decimals=1), np.round(dataset.train_x.shape[0] / 1000,decimals=1))
                with open(pp, "rb") as f:
                    data = pickle.load(f)
                variables = ['Age', 'Food Habits', 'Health', 'Mobility']
                list_vars_per_interv = [variables] * len(dataset.intervention)
            for ind_d, d in enumerate(variables):
                if baseline:
                    axs.flatten()[ind_d].bar(x=[0, 1], height=[baseline[interv_desc][d][0], baseline[interv_desc][d][1]],
                                             edgecolor='#747474', facecolor='#b2b2b2', width=0.15, label="CBN", alpha=0.9)
                # gt distribution
                if bnl_dataset:
                    hist_data = data[d]
                else:
                    hist_data = data[d[0]]
                weights = np.ones_like(hist_data)/len(hist_data)
                h = axs.flatten()[ind_d].hist(hist_data, facecolor=color_pairs[ind_d][0], edgecolor=color_pairs[ind_d][1], label="GT", weights=weights)

                # mpe
                if plot_mpe:
                    xc = np.round(mpe[0,ind_d], decimals=1)
                    axs.flatten()[ind_d].axvline(x=xc, label='CSPN Max = {}'.format(xc), c='red')
                # pdf
                #vals_x, vals_pdf = compute_pdf(list_vars_per_interv[ind].index(d), N, low, high, intervention=interv_desc, visualize=False)
                #axs.flatten()[ind_d].plot(vals_x, vals_pdf, label="CSPN learned PDF", color="black", linestyle='solid', linewidth=1.5)
                vals_pdf_mean, vals_pdf_std = mean_vals_per_interv[interv_desc][d]
                axs.flatten()[ind_d].plot(vals_x, vals_pdf_mean, 'k', label="i-SPN", color='#CC4F1B', linestyle='solid', linewidth=1.5, zorder=10)
                axs.flatten()[ind_d].fill_between(vals_x, vals_pdf_mean - vals_pdf_std, vals_pdf_mean + vals_pdf_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', zorder=9)
                # convex hull
                if plot_convex_hull:
                    points = np.vstack((vals_x,vals_pdf[:,0])).T
                    hull = ConvexHull(points)
                    for ind_s, simplex in enumerate(hull.simplices):
                        label = 'Convex Hull' if ind_s == 0 else None
                        axs.flatten()[ind_d].plot(points[simplex, 0], points[simplex, 1], 'k-.',label=label)

                if bnl_dataset:
                    axs.flatten()[ind_d].set_ylim(0, 1.3)
                    axs.flatten()[ind_d].set_xticks([0, 1])
                else:
                    axs.flatten()[ind_d].set_xlim(-20,100)
                    axs.flatten()[ind_d].set_ylim(0,0.325)
                if not show_axes_and_legend:
                    axs.flatten()[ind_d].axes.xaxis.set_ticklabels([])
                    axs.flatten()[ind_d].axes.yaxis.set_ticklabels([])
                else:
                    axs.flatten()[ind_d].set_title('{}'.format(d))
                    axs.flatten()[ind_d].legend(prop={'size':9})
            #plt.suptitle('Intervention: {}, sampled {} steps over({},{}), hist_n_bins={}, Train LL {:.2f}\n{}'.format(interv_desc,N,low,high, len(h[0]), np.round(loss_curve[-1],decimals=2), comment))
            plt.tight_layout()
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_loc = os.path.join(save_dir, "iSPN_int_{}.png".format(interv_desc))
                plt.savefig(save_loc)
                print('Saved @ {}'.format(save_loc))
            else:
                plt.show()
            plt.clf()
            plt.close()