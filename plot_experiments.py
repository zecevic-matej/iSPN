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

def convert_l_to_dict(l):
    d = {}
    for x in l:
        i = str(x['interv'])
        d.update({i: x['data']})
    return d

def normalize(x, a=None, b=None, low=None, high=None):
    if a is None and b is None:
        a = 0
        b = 0
    if low is None and high is None:
        low = np.min(x)
        high = np.max(x)
    return ((b-a)*((x - low) / (high - low))) + a

# collect all experiment paths and assert same number of experiments per method
iSPN_exp_paths = glob('./results/experiments/*iSPN*.pkl')
MDN_exp_paths = glob('./results/experiments/*MDN*.pkl')
MADE_exp_paths = glob('./results/experiments/*MADE*.pkl')
assert(len(iSPN_exp_paths) == len(MDN_exp_paths) == len(MADE_exp_paths))

# load all of them and other information
iSPN_data = {}
MDN_data = {}
MADE_data = {}
dataset_names = set()
for p in iSPN_exp_paths + MDN_exp_paths + MADE_exp_paths:
    with open(p, "rb") as f:
        dataset_name = os.path.basename(p).split("_")[0]
        dataset_names.add(dataset_name)
        data = pickle.load(f)
        data_dict = {dataset_name: data}
        if p in iSPN_exp_paths:
            iSPN_data.update(data_dict)
        elif p in MDN_exp_paths:
            MDN_data.update(data_dict)
        elif p in MADE_exp_paths:
            MADE_data.update(data_dict)
dataset_names = list(dataset_names)
interventions = {}
variables = {}
seeds = set()
# TODO: make this section generic
# assumes that each intervention is also available for all other methods
for ind_dn, dn in enumerate(dataset_names):
    l_seed = iSPN_data[dn].keys()
    for s in l_seed:
        seeds.add(s)
    interventions.update({dn: list(iSPN_data[dn][606].keys())})
    try:
        l_var = list(iSPN_data[dn][606]["None"].keys())
    except:
        l_var = list(iSPN_data[dn][606][None].keys())
    variables.update({dn: l_var})
seeds = list(seeds)


# prepare iSPN results which are mean/std plots instead of bar charts
mean_vals_per_data = {}
for ind_dn, dn in enumerate(dataset_names):
    mean_vals_per_interv = {}
    for ind_i, i in enumerate(interventions[dn]):
        mean_vals_per_var = {}
        for ind_v, v in enumerate(variables[dn]):
            vals_pdfs = []
            for ind_s, s in enumerate(seeds):
                vals_pdf = iSPN_data[dn][s][i][v][1][: ,0]
                vals_pdfs.append(vals_pdf)
            vals_pdfs = np.stack(vals_pdfs)
            mean_vals_per_var.update({v: (np.mean(vals_pdfs, axis=0), np.std(vals_pdfs, axis=0))})
        mean_vals_per_interv.update({i: mean_vals_per_var})
    mean_vals_per_data.update({dn: mean_vals_per_interv})

# TODO: make this section generic
# collect all data used for training for collecting the ground truth eventually
CH_data_paths = glob('./results/gt/causal_health*.pkl')
CH_data = {}
for p in CH_data_paths:
    with open(p, "rb") as f:
        i = p.split('intervention_')[1].split('_')[0]
        CH_data.update({i: pickle.load(f)})
asia_data_path = './results/gt/asia_uniform_interventions_N10000.pkl'
asia_data = load_data(asia_data_path)
asia_data = convert_l_to_dict(asia_data)
earthquake_data_path = './results/gt/earthquake_uniform_interventions_N10000.pkl'
earthquake_data = load_data(earthquake_data_path)
earthquake_data = convert_l_to_dict(earthquake_data)
cancer_data_path = './results/gt/cancer_uniform_interventions_N10000.pkl'
cancer_data = load_data(cancer_data_path)
cancer_data = convert_l_to_dict(cancer_data)
gt_dict = {
    "earthquake": earthquake_data,
    "asia": asia_data,
    "cancer": cancer_data,
    "CH": CH_data,
}

# blacklist
for dn in dataset_names:
    intervs = []
    for i in interventions[dn]:
        if i == "dysp" or i == "do(A)=U(A)":
            continue
        else:
            intervs.append(i)
    interventions.update({dn: intervs})

# normalize CH
for i in interventions["CH"]:
    for v in variables["CH"]:
        v = v[0]
        d = gt_dict["CH"][i][v]
        gt_dict["CH"][i].update({v: normalize(d, a=-0.2, b=1, low=-20, high=100)})


# plot all of it
show_n_vars = 4
show_n_interv = 2
fig, axs = plt.subplots(4,int(show_n_vars*show_n_interv),figsize=(13,8)) # 4 datasets x 4 variables * 2 interventions
for ind_dn, dn in enumerate(dataset_names):
    for ind_i, i in enumerate(interventions[dn][:show_n_interv]):
        for ind_v, v in enumerate(variables[dn][:show_n_vars]):
            a_y = ind_dn
            a_x = ind_i*show_n_vars+ind_v

            # Ground Truth
            if v not in gt_dict[dn][str(i)].keys():  # for the CH case
                v_alt = v[0]
            else:
                v_alt = v
            hist_data = gt_dict[dn][str(i)][v_alt]
            weights = np.ones_like(hist_data) / len(hist_data)
            axs[a_y, a_x].hist(hist_data, label="GT", weights=weights)

            # iSPN
            vals_x = iSPN_data[dn][606][i][v][0]
            vals_pdf_mean = mean_vals_per_data[dn][i][v][0]
            if dn == "CH":
                vals_x = normalize(vals_x, a=-0.2, b=1.)
            vals_pdf_std = mean_vals_per_data[dn][i][v][1]
            axs[a_y, a_x].plot(vals_x, vals_pdf_mean, 'k', label="i-SPN", color='#CC4F1B', linestyle='solid',
                                      linewidth=1.5, zorder=10)
            axs[a_y, a_x].fill_between(vals_x, vals_pdf_mean - vals_pdf_std, vals_pdf_mean + vals_pdf_std,
                                              alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', zorder=9)

            # MDN
            height = MDN_data[dn][606][i][v][1]
            pos = MDN_data[dn][606][i][v][0]
            pos_diff = pos[1]-pos[0]
            pos = np.array([pos[0]+i*pos_diff for i in range(len(height))])
            if dn != "CH":
                pos += 0.1
            axs[a_y, a_x].bar(x=pos, height=height, color="green", alpha=0.5, width=0.1)

            # MADE
            height = MADE_data[dn][606][i][v][1]
            pos = normalize(MADE_data[dn][606][i][v][0], a=-0.2, b=1, low=-20, high=100)
            pos_diff = pos[1]-pos[0]
            pos = np.array([pos[0]+i*pos_diff for i in range(len(height))])
            if dn != "CH":
                pos += 0.2
            axs[a_y, a_x].bar(x=pos, height=height, color="red", alpha=0.5, width=0.1)

            # turn of axes
            if dn == "CH":
                axs[a_y, a_x].set_xlim(0,1)
            axs[a_y, a_x].set_ylim(bottom=0)
            axs[a_y, a_x].axes.xaxis.set_ticklabels([])
            axs[a_y, a_x].axes.yaxis.set_ticklabels([])
            #axs[a_y, a_x].set_title(f"V:{v},I:{i},D:{dn}", fontsize=5)

plt.tight_layout()
plt.show()
plt.close()
plt.clf()


