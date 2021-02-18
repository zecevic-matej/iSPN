import numpy as np
import torch.nn as nn
import torch.optim as optim
import mdn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import seaborn as sns
sns.set_theme()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
import pickle
import os


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
        path_datasets = ("../causal-spn//datasets/data_for_uniform_interventions_continuous/",
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
        self.gs = {}
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
            self.gs.update({intervention: xs})
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        normalize = True
        if normalize:
            for d in range(ys_all.shape[1]):
                ys_all[:,d] = (ys_all[:,d] - np.min(ys_all[:,d])) / (np.max(ys_all[:,d]) - np.min(ys_all[:,d]))
        test_leftout = int(0.15 * len(random_sorting))
        assert(test_leftout <= len(random_sorting) * 0.2 and test_leftout > batch_size)
        self.train_x = xs_all[:len(random_sorting)-test_leftout,:]
        self.test_x = xs_all[len(random_sorting)-test_leftout:,:]
        self.train_y = ys_all[:len(random_sorting)-test_leftout,:]
        self.test_y = ys_all[len(random_sorting)-test_leftout:,:]
        self.train_x = self.train_x.astype(np.float32)
        self.train_y = self.train_y.astype(np.float32)
        self.test_x = self.test_x.astype(np.float32)
        self.test_y = self.test_y.astype(np.float32)
        self.description = 'Causal Health (Cont.)'
        #import pdb; pdb.set_trace()
    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, index):
        X = self.train_x[index,:]
        y = self.train_y[index,:]
        return X, y
class BnLearnDataset(torch.utils.data.Dataset):
    def __init__(self, bif, batch_size, description):
        p = '../causal-spn/datasets/other/benchmark_data_for_uniform_interventions/{}_uniform_interventions_N10000.pkl'.format(bif)
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.data = data
        self.intervention = []
        ys_all = []
        xs_all = []
        self.gs = {}
        for ind, dict_interv in enumerate(self.data):
            interv = dict_interv['interv']
            self.intervention.append(interv)
            ys = np.array(dict_interv['data'])
            g = np.array(dict_interv['adjmat'],dtype=float).flatten()
            xs = np.tile(g.flatten()[:, np.newaxis], len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
            self.gs.update({interv: xs})
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
        self.train_x = self.train_x.astype(np.float32)
        self.train_y = self.train_y.astype(np.float32)
        self.test_x = self.test_x.astype(np.float32)
        self.test_y = self.test_y.astype(np.float32)
        self.description = description

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, index):
        X = self.train_x[index,:]
        y = self.train_y[index,:]
        return X, y


bs = 200
x_dim = 16#64
y_dim = 4#8
episodes = 3
num_sample_runs = 10
name = 'CH'
if name in ["asia", "earthquake", "cancer"]:
    use_bnl_d = True
else:
    use_bnl_d = False
if use_bnl_d:
    bnl_d = BnLearnDataset(name, bs, f'{name} Bernoulli Discrete')
else:
    bnl_d = CausalHealthDataset(bs)
dataloader = DataLoader(bnl_d,batch_size=bs)
do_avg_distr = False
top_save_dir = f'MDN_results_{name}/'

vals_per_seed = {}
master_seeds = [606, 1011, 3004, 5555, 12096]
for cur_seed in master_seeds:

    np.random.seed(cur_seed)
    torch.manual_seed(cur_seed)

    save_dir = os.path.join(top_save_dir, 'MDN_results_{}_seed_{}'.format(name, cur_seed))

    model = nn.Sequential(
        nn.Linear(x_dim, 10),
        nn.Tanh(),
        mdn.MDN(10, y_dim, 30)
    )

    optimizer = optim.Adam(model.parameters())

    fig, axs = plt.subplots(1,2,figsize=(12,8))
    loss_curve = []
    try:
        for e in range(episodes):
            loss_curve_in_episode = []
            for ins,outs in dataloader:
                model.zero_grad()
                pi, sigma, mu = model(ins)
                loss = mdn.mdn_loss(pi, sigma, mu, outs) # this is NLL, therefore, can be infinite
                loss.backward()
                optimizer.step()
                print('Episode: {}     Loss: {}                   '.format(e, loss.item()), end='\r', flush=True)
                loss_curve_in_episode.append(loss.item())
            loss_curve.append(np.mean(loss_curve_in_episode))
    except Exception as e:
        print(e)
        continue

    colors = ['pink', 'blue', 'orange', 'lightgreen', 'yellow', 'red', 'cyan', 'purple'] # ASSUMES: NEVER more than 8 variables
    fig_len_x = 2
    fig_len_y = 4
    if use_bnl_d:
        list_vars_per_interv = [bnl_d.data[i]['data'].columns.tolist() for i in range(len(bnl_d.intervention))]
        low = -0.25
        high = 1.25
    else:
        low = 0
        high = 1
    vals_per_intervention = {}
    for ind, interv_desc in enumerate(bnl_d.intervention):
        fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(12, 10))
        comment = ''

        if not use_bnl_d:
            pp = '../causal-spn/datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_{}_N100000.pkl'.format(
                interv_desc)
            ppN = int(pp.split("_N")[1].split(".pkl")[0]) if interv_desc != "None" else int(
                pp.split("None")[1].split("_N")[1].split(".pkl")[0])
            comment = "Histograms on {} K samples, Training on {} K samples".format(np.round(ppN / 1000, decimals=1),
                                                                                    np.round(
                                                                                        bnl_d.train_x.shape[0] / 1000,
                                                                                        decimals=1))
            with open(pp, "rb") as f:
                data = pickle.load(f)
            variables = ['Age', 'Food Habits', 'Health', 'Mobility']
            list_vars_per_interv = [variables] * len(bnl_d.intervention)
        else:
            variables = list_vars_per_interv[0]  # assumes that all datasets have the same variables, makes sure that ordering is consistent
            data = bnl_d.data[ind]['data']
        try:
            ins_interv = torch.from_numpy(bnl_d.gs[interv_desc][:bs,:].astype(np.float32))
            pi, sigma, mu = model(ins_interv)
            samples = np.stack([mdn.sample(pi, sigma, mu).data.numpy() for x in range(num_sample_runs)]).reshape(-1, y_dim)
        except Exception as e:
            print(e)
            continue

        vals_per_var = {}
        for ind_d, d in enumerate(variables):

            # gt distribution
            if use_bnl_d:
                hist_data = data[d]
            else:
                hist_data = data[d[0]]
                normalize = True
                if normalize:
                    hist_data = (hist_data - np.min(hist_data)) / (np.max(hist_data) - np.min(hist_data))
            weights = np.ones_like(hist_data)/len(hist_data)
            h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d], label="GT", weights=weights, edgecolor=colors[ind_d], alpha=1)
            axs.flatten()[ind_d].set_title('{}'.format(d))
            axs.flatten()[ind_d].set_xlim(low, high)

            # mdn
            hist_data_mdn = samples[:,ind_d]
            weights_mdn = np.ones_like(hist_data_mdn)/len(hist_data_mdn)
            if use_bnl_d:
                h_mdn = axs.flatten()[ind_d].hist(hist_data_mdn,bins=2, label="MDN", weights=weights_mdn, alpha=0.4, color='black')
                vals_pdf = h_mdn[0]
                vals_x = np.array([0., 1.])
                axs.flatten()[ind_d].set_ylim(0,1)
            else:
                h_mdn = axs.flatten()[ind_d].hist(hist_data_mdn, label="MDN", weights=weights_mdn, alpha=0.4, color='black')
                vals_pdf = h_mdn[0]
                vals_x = h_mdn[1][:-1]
                axs.flatten()[ind_d].set_ylim(0,.4)

            axs.flatten()[ind_d].legend(prop={'size':9})

            vals_per_var.update({d: (vals_x, vals_pdf)})
        vals_per_intervention.update({interv_desc: vals_per_var})

        plt.suptitle('Intervention: {}, {} samples, Train NLL {:.2f}\n{}'.format(interv_desc,len(bnl_d), np.round(loss_curve[-1],decimals=2), comment))
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_loc = os.path.join(save_dir, "MDN_int_{}.png".format(interv_desc))
            plt.savefig(save_loc)
            print('Saved @ {}'.format(save_loc))
        else:
            plt.show()

    vals_per_seed.update({cur_seed: vals_per_intervention})


    # average distr.
    # given that mdn seems to be settling for a fixed rather than a dynamic conditional distribution
    if do_avg_distr:
        fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(12, 10))
        comment = ''

        ins_interv = [torch.from_numpy(bnl_d.gs[interv_desc][:bs, :].astype(np.float32)) for ind, interv_desc in enumerate(bnl_d.intervention)]
        samples_all_interv = []
        for ii in ins_interv:
            pi, sigma, mu = model(ii)
            samples = np.stack([mdn.sample(pi, sigma, mu).data.numpy() for x in range(num_sample_runs)]).reshape(-1, y_dim)
            samples_all_interv.append(samples)
        samples_all_interv = np.stack(samples_all_interv).reshape(-1,y_dim)

        data = np.stack([x['data'] for x in bnl_d.data]).reshape(-1,y_dim)
        variables = list_vars_per_interv[0]  # assumes that all datasets have the same variables, makes sure that ordering is consistent
        for ind_d, d in enumerate(variables):
            # gt distribution
            hist_data = data[:,ind_d]
            weights = np.ones_like(hist_data)/len(hist_data)
            h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d], label="GT", weights=weights, edgecolor=colors[ind_d], alpha=1)
            axs.flatten()[ind_d].set_title('{}'.format(d))
            axs.flatten()[ind_d].set_xlim(low, high)
            hist_data_mdn = samples[:,ind_d]
            weights_mdn = np.ones_like(hist_data_mdn)/len(hist_data_mdn)
            axs.flatten()[ind_d].hist(hist_data_mdn,bins=2, label="MDN", weights=weights_mdn, alpha=0.4, color='black')
            axs.flatten()[ind_d].legend(prop={'size':9})
            axs.flatten()[ind_d].set_ylim(0,1)
        plt.suptitle('Average Distribution across all Interv. Distr. (+Obs), Train NLL {:.2f}\n{}'.format(np.round(loss_curve[-1],decimals=2), comment))
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_loc = os.path.join(save_dir, "MDN_average_distr.png")
            plt.savefig(save_loc)
            print('Saved @ {}'.format(save_loc))
        else:
            plt.show()

    plt.close()
    plt.clf()

if top_save_dir:
    if not os.path.exists(top_save_dir):
        os.makedirs(top_save_dir)
    save_loc = os.path.join(top_save_dir, f"{name}_MDN_pdfs_per_seed.pkl")
    with open(save_loc, "wb") as f:
        pickle.dump(vals_per_seed, f)