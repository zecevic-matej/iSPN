import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import seaborn as sns
sns.set_theme()
import pickle
import torch
from models import MADE
from train import train_epoch, train_epoch_gaussian
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
        self.gs_ys = {}
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
            self.gs_ys.update({intervention: np.hstack((xs,ys))})
        random_sorting = np.random.permutation(sum([x.shape[0] for x in xs_all]))
        xs_all = np.vstack(xs_all)[random_sorting]
        ys_all = np.vstack(ys_all)[random_sorting]
        normalize = False
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
        #return X, y
        return np.hstack((X, y))  # y

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
        self.gs_ys = {}
        for ind, dict_interv in enumerate(self.data):
            interv = dict_interv['interv']
            self.intervention.append(interv)
            ys = np.array(dict_interv['data'])
            g = np.array(dict_interv['adjmat'],dtype=float).flatten()
            xs = np.tile(g.flatten()[:, np.newaxis], len(ys)).T
            xs_all.append(xs)
            ys_all.append(ys)
            self.gs.update({interv: xs})
            self.gs_ys.update({interv: np.hstack((xs,ys))})
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
        return np.hstack((X,y)) # y


bs = 200
x_dim = 16 + 4 #25 + 5#64 + 8
y_dim = 4#5#8 # this is an Autoencoder, thus, 'unsupervised' and not really used except during sampling
epochs = 10#30
num_sample_runs = 10
name = 'CH'
if name in ["asia", "earthquake", "cancer"]:
    use_bnl_d = True
else:
    use_bnl_d = False
if use_bnl_d:
    bnl_d = BnLearnDataset(name, bs, f'{name} Bernoulli Discrete')
    gaussian = False
else:
    bnl_d = CausalHealthDataset(bs)
    gaussian = True
dataloader = DataLoader(bnl_d,batch_size=bs)

top_save_dir = f'MADE_results_{name}/'

vals_per_seed = {}
master_seeds = [606, 1011, 3004, 5555, 12096]
for cur_seed in master_seeds:

    np.random.seed(cur_seed)
    torch.manual_seed(cur_seed)

    save_dir = os.path.join(top_save_dir, 'MADE_results_{}_seed_{}'.format(name, cur_seed))

    # Define model, optimizer, and scheduler.
    hidden_dims = [256]
    model = MADE(n_in=x_dim, hidden_dims=hidden_dims, random_order=False, seed=cur_seed, gaussian=gaussian)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Training loop.
    for epoch in range(1, epochs + 1):
        if gaussian:
            loss = train_epoch_gaussian(model, dataloader, epoch, optimizer)
        else:
            loss = train_epoch(model, dataloader, epoch, optimizer, scheduler=scheduler)


    #choices = np.random.randint(0,len(bnl_d),bs)
    #samples = torch.from_numpy(bnl_d[choices].astype(np.float32))

    colors = ['pink', 'blue', 'orange', 'lightgreen', 'yellow', 'red', 'cyan', 'purple'] # ASSUMES: NEVER more than 8 variables
    fig_len_x = 2
    fig_len_y = 4
    if use_bnl_d:
        list_vars_per_interv = [bnl_d.data[i]['data'].columns.tolist() for i in range(len(bnl_d.intervention))]
        low = -0.25
        high = 1.25
    else:
        low = -20#0
        high = 100#1
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

        ins_interv = torch.from_numpy(bnl_d.gs[interv_desc][:bs,:].astype(np.float32))
        samples = torch.from_numpy(bnl_d.gs_ys[interv_desc][:bs,:].astype(np.float32))
        # sample the first dimension of each vector
        for _, dim in enumerate(range(x_dim-y_dim,x_dim)):
            output = model(samples)
            if gaussian:
                gauss = torch.distributions.Normal(output[:,dim],1)
                sample_output = gauss.sample()
            else:
                bernoulli = torch.distributions.Bernoulli(output[:, dim])
                sample_output = bernoulli.sample()
            samples[:, dim] = sample_output

        vals_per_var = {}
        for ind_d, d in enumerate(variables):

            # gt distribution
            if use_bnl_d:
                hist_data = data[d]
            else:
                hist_data = data[d[0]]
                normalize = False
                if normalize:
                    hist_data = (hist_data - np.min(hist_data)) / (np.max(hist_data) - np.min(hist_data))

            weights = np.ones_like(hist_data)/len(hist_data)
            h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d], label="GT", weights=weights, edgecolor=colors[ind_d], alpha=1)
            axs.flatten()[ind_d].set_title('{}'.format(d))
            axs.flatten()[ind_d].set_xlim(low, high)

            # made
            hist_data_made = np.array(samples[:,x_dim-y_dim+ind_d])
            weights_made = np.ones_like(hist_data_made)/len(hist_data_made)
            if use_bnl_d:
                h_made = axs.flatten()[ind_d].hist(hist_data_made, bins=2, label="MADE", weights=weights_made, alpha=0.4, color='black')
                vals_pdf = h_made[0]
                vals_x = np.array([0., 1.])
                axs.flatten()[ind_d].set_ylim(0, 1)
            else:
                h_made = axs.flatten()[ind_d].hist(hist_data_made, label="MADE", weights=weights_made,
                                                   alpha=0.4, color='black')
                vals_pdf = h_made[0]
                vals_x = h_made[1][:-1]
                axs.flatten()[ind_d].set_ylim(0, .4)

            axs.flatten()[ind_d].legend(prop={'size':9})

            vals_per_var.update({d: (vals_x, vals_pdf)})
        vals_per_intervention.update({interv_desc: vals_per_var})

        plt.suptitle('Intervention: {}, {} samples, Train NLL {:.2f}\n{}'.format(interv_desc,len(bnl_d), np.round(loss,decimals=2), comment))
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_loc = os.path.join(save_dir, "MADE_int_{}.png".format(interv_desc))
            plt.savefig(save_loc)
            print('Saved @ {}'.format(save_loc))
        else:
            plt.show()

    vals_per_seed.update({cur_seed: vals_per_intervention})
    plt.close()
    plt.clf()

if top_save_dir:
    if not os.path.exists(top_save_dir):
        os.makedirs(top_save_dir)
    save_loc = os.path.join(top_save_dir, f"{name}_MADE_pdfs_per_seed.pkl")
    with open(save_loc, "wb") as f:
        pickle.dump(vals_per_seed, f)
