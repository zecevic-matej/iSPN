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
from glob import glob

class BnLearnDataset():
    def __init__(self, bif, batch_size, description, whitelist):
        N = 10000
        l_data_paths = glob(f"../ate_benchmark/data/{bif}_interventions_N{N}.pkl")
        d = {}
        for p in l_data_paths:
            with open(p, 'rb') as f:
                data = pickle.load(f)
        for d in data:
            if 'interv_type' in d.keys():
                d['interv'] += f'_{d["interv_type"]}'
        self.data = data
        self.intervention = []
        ys_all = []
        xs_all = []
        for ind, dict_interv in enumerate(self.data):
            interv = dict_interv['interv']
            self.intervention.append(interv)
            ys = np.array(dict_interv['data'])
            g = np.array(dict_interv['adjmat'],dtype=float)
            # the first self-loop which by definition is empty (because DAG)
            # will be used as indicator between two different interventions
            # on same node
            if interv is not None and "zero" in interv:
                g[0,0] = -10
            elif interv is not None and "one" in interv:
                g[0,0] = 10
            g = g.flatten()
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

class KidneyStoneDataset():
    def __init__(self, batch_size, description):
        import pandas as pd
        # creating Kidney Stone example data
        D = pd.DataFrame(np.zeros((700, 3)), columns=['T', 'Z', 'R'])
        A_obs = pd.DataFrame(np.array([[False, False, True], [True, False, True], [False, False, False]]),
                         columns=['T', 'Z', 'R'], index=['T', 'Z', 'R'])
        A_int = pd.DataFrame(np.array([[False, False, True], [False, False, True], [False, False, False]]),
                         columns=['T', 'Z', 'R'], index=['T', 'Z', 'R'])
        D.loc[:562 - 1, 'R'] = 1
        # D.loc[273:562+61-1,'T'] = 0
        D.loc[:273 - 1, 'T'] = 1
        D.loc[562 + 61:, 'T'] = 1
        D.loc[81:273 - 1, 'Z'] = 1
        D.loc[562 + 61 + 6:, 'Z'] = 1
        D.loc[273 + 234:562 - 1, 'Z'] = 1
        D.loc[562 + 36:562 + 61 - 1, 'Z'] = 1
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 1)]) == 273)
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 0)]) == 289)
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 1) & (D['Z'] == 0)]) == 81)
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 1) & (D['Z'] == 1)]) == 192)
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 0) & (D['Z'] == 0)]) == 234)
        assert (len(D.loc[(D['R'] == 1) & (D['T'] == 0) & (D['Z'] == 1)]) == 55)
        print(f'\n++++\nKidney Stone (Simpson Paradox) Toy Dataset with Sample Size N={len(D)}')
        data = []
        data.append({'data': D.loc[(D['T'] == 1.)], 'adjmat': A_int, 'interv': 'T_zero'})
        data.append({'data': D.loc[(D['T'] == 0.)], 'adjmat': A_int, 'interv': 'T_one'}) # naming must be flipped here, otherwise does not make sense
        self.data = data
        self.intervention = []
        ys_all = []
        xs_all = []
        for ind, dict_interv in enumerate(self.data):
            interv = dict_interv['interv']
            self.intervention.append(interv)
            ys = np.array(dict_interv['data'])
            g = np.array(dict_interv['adjmat'],dtype=float)
            if interv is not None and "zero" in interv:
                g[0,0] = -10
            elif interv is not None and "one" in interv:
                g[0,0] = 10
            g = g.flatten()
            xs = np.tile(g.flatten()[:, np.newaxis], len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
        np.random.seed(0)
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        test_leftout = int(0.05 * len(random_sorting))
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


    # conf.num_epochs = 70
    # conf.batch_size = 100
    # num_sum_weights = 2400
    # num_leaf_weights = 96
    # use_simple_mlp = True
    # bnl_dataset = "asia" # 'earthquake'
    # whitelist = None
    # dataset_name = bnl_dataset
    # description = '{} (bnlearn, Bernoulli)'.format(bnl_dataset)
    # dataset = BnLearnDataset(bnl_dataset, conf.batch_size, description, whitelist)
    # #tr_ef_pair = [('Burglary', 'Alarm')]
    # tr_ef_pair = [('asia','tub'),('bronc', 'dysp')]

    conf.num_epochs = 70
    conf.batch_size = 30
    num_sum_weights = 2400
    num_leaf_weights = 96
    use_simple_mlp = True
    bnl_dataset = 'kidney'
    whitelist = None
    dataset_name = bnl_dataset
    description = 'kidney stone simpson paradox'
    dataset = KidneyStoneDataset(conf.batch_size, description)
    tr_ef_pair = [('T', 'R')]

    #low=-0.5
    #high=1.5
    low=-1.5
    high=2.5

    save_dir=f"../ate_benchmark/iSPN_trained_uniform_interventions_{dataset_name}_ep{conf.num_epochs}/" # if not specified, then plots are plotted instead of saved

    plot_loss_curve = False
    plot_mpe = False
    plot_convex_hull = False
    show_axes_and_legend = True

    without_training_from_file = None #'appendix_ablation/iSPN_trained_uniform_interventions_CH/iSPN_pdfs_per_seed.pkl'

    #################################################################

    baseline = None #'../results/baseline_CBN.pkl' # Causal Baseline, used for Motivation Figure

    master_seeds = [606, 1011, 3004, 5555, 12096]

    batch_size = conf.batch_size
    if without_training_from_file is None:
        vals_per_seed = {}
        loss_curves_per_seed = []
        for cur_seed in master_seeds:

            np.random.seed(cur_seed)
            tf.set_random_seed(cur_seed)

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

            if plot_loss_curve:
                plt.plot(range(len(loss_curve)), loss_curve)
                plt.title('Intervention: {}\nTraining Gaussian CSPN on {} with {} Samples'.format(dataset.intervention, dataset.description, len(dataset.train_x)))
                plt.ylabel('Log-Likelihood')
                plt.xlabel('Epoch #')
                plt.show()

            def compute_pdf(dim, N, low, high, intervention, visualize=False, dd=None, comment=None):
                if bnl_dataset:
                    dict_interv = next(item for item in dataset.data if item["interv"] == intervention)
                    g = np.array(dict_interv['adjmat'],dtype=float)
                    if dict_interv['interv'] is not None and "zero" in dict_interv['interv']:
                        g[0, 0] = -10
                    elif dict_interv['interv'] is not None and "one" in dict_interv['interv']:
                        g[0, 0] = 10
                    g = g.flatten()
                else:
                    pass
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
            vals_per_seed.update({cur_seed: vals_per_intervention})

            loss_curves_per_seed.append(loss_curve)

            # clear the session, to re-run
            tf.reset_default_graph()


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
    plt.plot(range(1,len(mean_loss_curve)+1), mean_loss_curve, color='#267a85', linestyle='solid', linewidth=1.5)
    plt.fill_between(range(1,len(mean_loss_curve)+1), mean_loss_curve - std_loss_curve, mean_loss_curve + std_loss_curve, alpha=0.5, edgecolor='#267a85', facecolor='#37b4c7')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "iSPN_mean_loss_curve.png"))
    plt.clf()
    plt.close()



    import scipy.integrate as integrate
    import scipy.special as special
    def compute_expectation_via_integration(pdf_vals, low, high, N):
        sampling_locs = np.linspace(low, high, N)
        def closest_neighbor_pdf(pdf_vals, sampling_locs, x):
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx
            idx = find_nearest(sampling_locs, x)
            return pdf_vals[idx]

        expectation_integrand = lambda x: x * closest_neighbor_pdf(pdf_vals, sampling_locs, x)
        return integrate.quad(expectation_integrand, low, high)[0]

    def compute_ATE(pdf_vals_tr_one, pdf_vals_tr_zero, low, high, N):
        # idea 1
        #from scipy.special import softmax
        #return (softmax(pdf_vals_tr_one) * sampling_locs).sum() - (softmax(pdf_vals_tr_zero) * sampling_locs).sum()

        # idea 2
        return compute_expectation_via_integration(pdf_vals_tr_one, low, high, N) - compute_expectation_via_integration(pdf_vals_tr_zero, low, high, N)


    import warnings
    warnings.filterwarnings("ignore")
    plot_all_seeds_separately = True
    plot_also_means = True # if above is true, then this ensures that we also plot the mean one
    plot_expectation = True
    seeds_considered = master_seeds.copy()
    if plot_also_means:
        seeds_considered += [master_seeds]

    for cur_s_separate in seeds_considered:

        if plot_all_seeds_separately:
            if isinstance(cur_s_separate, list):
                cur_msl = cur_s_separate
            else:
                cur_msl = [cur_s_separate]
            seed_save_dir = os.path.join(save_dir, f"seed_{cur_s_separate}")
        else:
            cur_msl = master_seeds

        # vals_x assumed to be same everywhere and not collected here
        #vals_x = vals_per_seed[master_seeds[0]][dataset.intervention[0]][...]
        mean_vals_per_interv = {}
        for ind, interv_desc in enumerate(dataset.intervention):
            mean_vals_per_var = {}
            for v in vals_per_seed[cur_msl[0]][interv_desc].keys():
                vals_pdfs = []
                for seed in cur_msl:
                    vals_pdf = vals_per_seed[seed][interv_desc][v][1][:,0]
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

                if interv_desc is not None:
                    part_of_tr_ef_pairs = [interv_desc.split("_")[0] in x for x in tr_ef_pair]
                    idx = np.argmax(part_of_tr_ef_pairs)
                    tr, ef = tr_ef_pair[idx]
                    if plot_expectation and tr_ef_pair is not None and any(part_of_tr_ef_pairs) and d == ef:
                        exp_int = compute_expectation_via_integration(vals_pdf_mean, low, high, N)
                        axs.flatten()[ind_d].axvline(x=exp_int, c='red')

                # convex hull
                if plot_convex_hull:
                    points = np.vstack((vals_x,vals_pdf[:,0])).T
                    hull = ConvexHull(points)
                    for ind_s, simplex in enumerate(hull.simplices):
                        label = 'Convex Hull' if ind_s == 0 else None
                        axs.flatten()[ind_d].plot(points[simplex, 0], points[simplex, 1], 'k-.',label=label)

                if bnl_dataset:
                    axs.flatten()[ind_d].set_ylim(0, 1.3)
                    axs.flatten()[ind_d].set_xlim(-0.5, 1.5)
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
            if interv_desc is not None:
                ate = compute_ATE(mean_vals_per_interv[tr+"_one"][ef][0], mean_vals_per_interv[tr+"_zero"][ef][0], low, high, N)
                plt.suptitle(f'ATE(Tr={tr} Ef={ef}) = {ate:.4f} \t E[Ef]={exp_int:.4f}')
            plt.tight_layout()
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if plot_all_seeds_separately:
                    if not os.path.exists(seed_save_dir):
                        os.makedirs(seed_save_dir)
                    save_loc = os.path.join(seed_save_dir, "iSPN_seed_{}_int_{}.png".format(cur_s_separate, interv_desc))
                else:
                    save_loc = os.path.join(save_dir, "iSPN_int_{}.png".format(interv_desc))
                plt.savefig(save_loc)
                print('Saved @ {}'.format(save_loc))
            else:
                plt.show()
            plt.clf()
            plt.close()

        if not plot_all_seeds_separately:
            break