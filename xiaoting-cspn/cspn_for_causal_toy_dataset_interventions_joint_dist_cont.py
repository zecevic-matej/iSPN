import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
np.set_printoptions(suppress=True)

def get_graph_for_intervention(intervention):
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

class CausalDataset():
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
            g = get_graph_for_intervention(intervention)
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
        #import pdb; pdb.set_trace()

with tf.device('/GPU:0'):
    conf = Config()
    conf.model_name = 'cspn'
    conf.num_epochs = 130
    conf.batch_size = 1000

    batch_size = conf.batch_size
    x_shape = (batch_size, 16)
    y_shape = (batch_size, 4)
    x_dims = 16
    y_dims = 4

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
    args.num_gauss = 4
    args.dist = 'Gauss'
    spn = RAT_SPN.RatSpn(num_classes=1, region_graph=rg, name="spn", args=args)
    print("TOTAL", spn.num_params())

    dataset = CausalDataset(batch_size)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    trainer = CspnTrainer(spn, dataset, x_ph, train_ph, conf, sess=sess)
    loss_curve = trainer.run_training(no_save=True)

    # TODO: this is not implemented by xshao and others
    #samples = trainer.spn.sample()

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

    save_loss = False
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.title('Intervention: {}\nTraining Gaussian CSPN on Causal Health (Cont.) with {} Samples'.format(dataset.intervention, len(dataset.train_x)))
    plt.ylabel('Log-Likelihood')
    plt.xlabel('Epoch #')
    if save_loss:
        plt.show()


    def compute_pdf(dim, N, low, high, intervention, visualize=False, dd=None, comment=None):
        g = get_graph_for_intervention(intervention)
        y_batch = np.zeros((N,4))
        y_batch[:,dim] = np.linspace(low,high,N)
        marginalized = np.ones((N,4))
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

    # # params
    # N = 1000
    # dd = {0: "Age", 1: "Food Habits", 2: "Health", 3: "Mobility"}
    # dim = 2  # checking Food Habits, (A,F,H,M)=(0,1,2,3)
    # low=-20
    # high=100
    # comment = "GT is a Gaussian (-5,0.1)"
    # # computation
    # pdf_vals = compute_pdf(dim,N,low,high,visualize=True,dd=dd,comment=comment)

    colors = ['pink', 'blue', 'orange', 'lightgreen']
    N = batch_size
    low=-20
    high=100
    save_dir="iSPN_trained_uniform_interventions/"
    for interv_desc in dataset.intervention:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # sampling density #plt.plot(np.linspace(low,high,N),np.zeros(N), marker="o"); plt.show()
        pp = '../datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_{}_N100000.pkl'.format(interv_desc)
        ppN = int(pp.split("_N")[1].split(".pkl")[0]) if interv_desc != "None" else int(pp.split("None")[1].split("_N")[1].split(".pkl")[0])
        comment = "Histograms on {} K samples, Training on {} K samples".format(np.round(ppN/ 1000,decimals=1), np.round(dataset.train_x.shape[0] / 1000,decimals=1))
        with open(pp, "rb") as f:
            data = pickle.load(f)
        for ind_d, d in enumerate(['Age', 'Food Habits', 'Health', 'Mobility']):
            # gt distribution
            hist_data = data[d[0]]#dataset.data[d[0]]
            weights = np.ones_like(hist_data)/len(hist_data)
            h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d],bins=15, label="GT", weights=weights, edgecolor=colors[ind_d])
            axs.flatten()[ind_d].set_title('{}'.format(d))
            axs.flatten()[ind_d].set_xlim(low, high)
            # mpe
            xc = np.round(mpe[0,ind_d], decimals=1)
            axs.flatten()[ind_d].axvline(x=xc, label='CSPN Max = {}'.format(xc), c='red')
            # pdf
            vals_x, vals_pdf = compute_pdf(ind_d, N, low, high, intervention=interv_desc, visualize=False)
            axs.flatten()[ind_d].plot(vals_x, vals_pdf, label="CSPN learned PDF", color="black", linestyle='solid', linewidth=1.5)
            axs.flatten()[ind_d].legend(prop={'size':9})
        plt.suptitle('Intervention: {}, sampled {} steps over({},{}), hist_n_bins={}, Train LL {:.2f}\n{}'.format(interv_desc,N,low,high, len(h[0]), np.round(loss_curve[-1],decimals=2), comment))
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "iSPN_int_{}.png".format(interv_desc)))
        else:
            plt.show()