import pickle
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def load_data(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d

p = glob('/Users/mzecevic/Desktop/repos/causal-spn/xiaoting-cspn/appendix_ablation/proper_run_ep70_100k/*/*.pkl')
d = {}
for x in p:
    d.update({x.split("_CH_")[1].split("_ep")[0]: load_data(x)})
lc = {}
for ind, k in enumerate(d.keys()):
    if 'loss curve' in d[k].keys():
        lc.update({k: d[k]['loss curve']})

all_seeds = list(d['50'].keys())[:-1]
vals_x = d['50'][606]['None']['Age'][0]

# vals_x assumed to be same everywhere and not collected here
# vals_x = vals_per_seed[master_seeds[0]][dataset.intervention[0]][...]
d_m = {}
for k in d.keys():
    vals_per_seed = d[k]
    all_interv = list(vals_per_seed[606].keys())
    mean_vals_per_interv = {}
    for ind, interv_desc in enumerate(all_interv):
        mean_vals_per_var = {}
        for v in vals_per_seed[all_seeds[0]][interv_desc].keys():
            vals_pdfs = []
            for seed in all_seeds:
                vals_pdf = vals_per_seed[seed][interv_desc][v][1][:, 0]
                vals_pdfs.append(vals_pdf)
            vals_pdfs = np.stack(vals_pdfs)
            mean_vals_per_var.update({v: (np.mean(vals_pdfs, axis=0), np.std(vals_pdfs, axis=0))})
        mean_vals_per_interv.update({interv_desc: mean_vals_per_var})
    d_m.update({k: mean_vals_per_interv})

save_dir = '/Users/mzecevic/Desktop/repos/causal-spn/xiaoting-cspn/appendix_ablation/proper_run_ep70_100k/'

print('Assuming Causal Health Dataset.')
fig_len_x = len(d_m.keys())
fig_len_y = 4
color_pairs = [
    ('#ffcfe2', '#ffbed7'),
    ('#ffff6b', '#d1d100'),
    ('#88ed83', '#4fe547'),
    ('#7fccf4', '#40b3ef'),
]
share_x = True
share_y = False
show_axes_and_legend = False

fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(13, 12), sharey=share_y, sharex=share_x)
for ind, k in enumerate([list(d_m.keys())[x] for x  in [2,0,3,5,1,4]]):
    pp = f'/Users/mzecevic/Desktop/repos/causal-spn/datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_do(H)={k}_N100000.pkl'
    with open(pp, "rb") as f:
        data = pickle.load(f)
    variables = ['Age', 'Food Habits', 'Health', 'Mobility']
    for ind_d, d in enumerate(variables):
        hist_data = data[d[0]]
        weights = np.ones_like(hist_data) / len(hist_data)
        h = axs[ind, ind_d].hist(hist_data, facecolor=color_pairs[ind_d][0], edgecolor=color_pairs[ind_d][1],
                                      label="GT", weights=weights)

        # pdf
        vals_pdf_mean, vals_pdf_std = d_m[k][f'do(H)={k}'][d]
        axs[ind, ind_d].plot(vals_x, vals_pdf_mean, 'k', label="i-SPN", color='#CC4F1B', linestyle='solid',
                                  linewidth=1.5, zorder=10)
        axs[ind, ind_d].fill_between(vals_x, vals_pdf_mean - vals_pdf_std, vals_pdf_mean + vals_pdf_std, alpha=0.5,
                                          edgecolor='#CC4F1B', facecolor='#FF9848', zorder=9)

        axs[ind, ind_d].set_xlim(-20, 100)
        axs[ind, ind_d].set_ylim(0, 0.4)
        if not show_axes_and_legend:
            axs[ind, ind_d].axes.xaxis.set_ticklabels([])
            axs[ind, ind_d].axes.yaxis.set_ticklabels([])
        else:
            axs[ind, ind_d].set_title('{}'.format(d))
            axs[ind, ind_d].legend(prop={'size': 9})
plt.subplots_adjust(wspace=0, hspace=0.1)
#plt.tight_layout()
if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loc = os.path.join(save_dir, f"i-SPN_ablation_interventions.png")
    plt.savefig(save_loc)
    print('Saved @ {}'.format(save_loc))
else:
    plt.show()
plt.clf()
plt.close()





# loss curves
for k in lc.keys():
    m,s = lc[k]
    plt.plot(range(len(m)), m, label=f"{k}")
    plt.fill_between(range(len(m)), m-s,m+s)
plt.legend()
plt.show()
plt.clf()
plt.close()
















# single best vs mean
p = '/Users/mzecevic/Desktop/repos/causal-spn/xiaoting-cspn/appendix_ablation/iSPN_trained_uniform_interventions_CH_50_ep50/iSPN_pdfs_per_seed.pkl'
vals_per_seed = load_data(p)
all_interv = list(vals_per_seed[606].keys())
mean_vals_per_interv = {}
for ind, interv_desc in enumerate(all_interv):
    mean_vals_per_var = {}
    for v in vals_per_seed[all_seeds[0]][interv_desc].keys():
        vals_pdfs = []
        for seed in all_seeds:
            vals_pdf = vals_per_seed[seed][interv_desc][v][1][:, 0]
            vals_pdfs.append(vals_pdf)
        vals_pdfs = np.stack(vals_pdfs)
        mean_vals_per_var.update({v: (np.mean(vals_pdfs, axis=0), np.std(vals_pdfs, axis=0))})
    mean_vals_per_interv.update({interv_desc: mean_vals_per_var})

fig, axs = plt.subplots(2, 4, figsize=(13, 12), sharey=share_y, sharex=share_x)
pp = f'/Users/mzecevic/Desktop/repos/causal-spn/datasets/data_for_uniform_interventions_continuous/causal_health_toy_data_continuous_intervention_do(H)=50_N100000.pkl'
with open(pp, "rb") as f:
    data = pickle.load(f)
variables = ['Age', 'Food Habits', 'Health', 'Mobility']
for ind in range(2):
    if ind == 0:
        gg = vals_per_seed[606]
    else:
        gg = mean_vals_per_interv
    for ind_d, d in enumerate(variables):
        hist_data = data[d[0]]
        weights = np.ones_like(hist_data) / len(hist_data)
        h = axs[ind, ind_d].hist(hist_data, facecolor=color_pairs[ind_d][0], edgecolor=color_pairs[ind_d][1],
                                      label="GT", weights=weights)

        # pdf
        if ind == 1:
            vals_pdf_mean, vals_pdf_std = gg['do(H)=50'][d]
            axs[ind, ind_d].plot(vals_x, vals_pdf_mean, 'k', label="i-SPN", color='#CC4F1B', linestyle='solid',
                                  linewidth=1.5, zorder=10)
            axs[ind, ind_d].fill_between(vals_x, vals_pdf_mean - vals_pdf_std, vals_pdf_mean + vals_pdf_std, alpha=0.5,
                                              edgecolor='#CC4F1B', facecolor='#FF9848', zorder=9)
        else:
            axs[ind, ind_d].plot(vals_x, gg['do(H)=50'][d][1], 'k', label="i-SPN", color='#CC4F1B', linestyle='solid',
                                  linewidth=1.5, zorder=10)

        axs[ind, ind_d].set_xlim(-20, 100)
        axs[ind, ind_d].set_ylim(0, 0.4)
        if not show_axes_and_legend:
            axs[ind, ind_d].axes.xaxis.set_ticklabels([])
            axs[ind, ind_d].axes.yaxis.set_ticklabels([])
        else:
            axs[ind, ind_d].set_title('{}'.format(d))
            axs[ind, ind_d].legend(prop={'size': 9})
plt.subplots_adjust(wspace=0, hspace=0.1)
#plt.tight_layout()
if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loc = os.path.join(save_dir, f"i-SPN_ablation_mean_vs_single.png")
    plt.savefig(save_loc)
    print('Saved @ {}'.format(save_loc))
else:
    plt.show()
plt.clf()
plt.close()