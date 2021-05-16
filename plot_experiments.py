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
show_n_interv = 4
fig, axs = plt.subplots(4,int(show_n_vars*show_n_interv),figsize=(13,9)) # 4 datasets x 4 variables * 2 interventions
general_bar_width = 0.2
centering_offset = general_bar_width
plot_with_info = True
plot_compact = True
ordering = [3,1,0,2] # reordering dataset by liking
specific_variables = {
    'asia': ['xray','tub', 'lung','either'],
    'earthquake': ['Burglary', 'Earthquake', 'Alarm', 'MaryCalls'],
    'cancer': ['Smoker', 'Pollution', 'Cancer', 'Xray'],
                      }
if specific_variables:
    for dn in dataset_names:
        if dn in specific_variables.keys():
            assert(len(specific_variables[dn]) == show_n_vars)
d_jsds = {}
for gg in ['tub', 'Cancer', 'Alarm', 'do(H)=U(H)']:
    jsds = []
    for ind_dn, dn in enumerate([dataset_names[i] for i in ordering]):
        print(f'Plotting for {dn} the Interventions {interventions[dn][:show_n_interv]} using Variables {variables[dn][:show_n_vars]} (not showing specific variables)')
        for ind_i, i in enumerate(interventions[dn][:show_n_interv]):
            for ind_v, v in enumerate(variables[dn][:show_n_vars]):
                a_y = ind_dn
                a_x = ind_i*show_n_vars+ind_v
                if specific_variables and dn in specific_variables.keys():
                    v = specific_variables[dn][ind_v]

                # Ground Truth
                if v not in gt_dict[dn][str(i)].keys():  # for the CH case
                    v_alt = v[0]
                else:
                    v_alt = v
                hist_data = gt_dict[dn][str(i)][v_alt]
                weights = np.ones_like(hist_data) / len(hist_data)
                if dn == 'CH':
                    gt_hist = axs[a_y, a_x].hist(hist_data, label="GT", weights=weights, color='black')
                else:
                    h_bins = [0., 0.5, 1.]
                    h, e = np.histogram(hist_data, bins=h_bins, weights=weights)
                    axs[a_y, a_x].bar(np.array([0.,1.])+general_bar_width-centering_offset, h, width=general_bar_width, color='black', linewidth=0)

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
                height_mdn = MDN_data[dn][606][i][v][1]
                if dn == 'CH':
                    pos = MDN_data[dn][606][i][v][0]
                    pos_diff = pos[1]-pos[0]
                    axs[a_y, a_x].bar(x=pos, height=height_mdn, color="#65cc1b", alpha=0.5, width=pos_diff, linewidth=0)
                else:
                    axs[a_y, a_x].bar(x=np.array([0.,1.])-centering_offset, height=height_mdn, color="#65cc1b", alpha=0.5, width=general_bar_width, linewidth=0)

                # MADE
                height_made = MADE_data[dn][606][i][v][1]
                if dn == 'CH':
                    pos = normalize(MADE_data[dn][606][i][v][0], a=-0.2, b=1, low=-20, high=100)
                    pos_diff = pos[1]-pos[0]
                    axs[a_y, a_x].bar(x=pos, height=height_made, color="#821bcc", alpha=0.5, width=pos_diff, linewidth=0)
                else:
                    axs[a_y, a_x].bar(x=np.array([0.,1.])+2*general_bar_width-centering_offset, height=height_made, color="#821bcc", alpha=0.5, width=general_bar_width, linewidth=0)

                if i == gg:#'tub':
                    # compute JSD for the ASIA dataset for Intervention do(tub=B(0.5))
                    kl = lambda p, q: p*(np.log(p/q)) if q>0 and p>0 else 0 # to avoid 0 case
                    # sanity check
                    assert(np.allclose(sum([kl(h[i],h[i]) for i in range(2)]), 0.))
                    # jsd gt vs mdn
                    def compute_jsd(p, q):
                        kl_pq = sum([kl(p[i],q[i]) for i in range(2)]) # assumes binary distribution
                        kl_qp = sum([kl(q[i], p[i]) for i in range(2)])
                        jsd = 0.5 * kl_pq + 0.5 * kl_qp
                        return jsd
                    if dn == "CH":
                        h = gt_hist[0]
                    jsd_mdn = compute_jsd(h, height_mdn)
                    print(f'JSD(GT, MDN; Intervention = {i}, Variable = {v}) = {jsd_mdn:.4f}   [{h} vs. {height_mdn}]')
                    # jsd gt vs made
                    jsd_made = compute_jsd(h, height_made)
                    print(f'JSD(GT, MADE; Intervention = {i}, Variable = {v}) = {jsd_made:.4f}   [{h} vs. {height_made}]')
                    # jsd gt vs iSPN
                    distr = np.array([vals_pdf_mean[24], vals_pdf_mean[74]]) # known in this case, simply the "pdf" values
                    # given that for the binary ones we don't get a proper PDF, we have to "normalize" somehow
                    #if gg == "do(M)=U(M)":
                    #    import pdb; pdb.set_trace()
                    if dn == "CH":
                        # in the case of Causal Health we do moment matching
                        normalized_distr = np.array([0., np.max(vals_pdf_mean)])
                        h = np.array([0., np.max(h)])
                    else:
                        diff = distr[0] - distr[1]
                        diff_from_one = distr[0] + distr[1] - 1
                        if diff > 0:
                            if abs(diff) > 0.5:
                                normalized_distr = np.array([1.,0.])
                            else:
                                normalized_distr = np.array([distr[0]-diff_from_one/2, distr[1]-diff_from_one/2]) #np.array([distr[0]-abs(diff)/2, distr[1]-abs(diff)/2]) #np.array([0.5+abs(diff), 0.5-abs(diff)])
                        else:
                            if abs(diff) > 0.5:
                                normalized_distr = np.array([0.,1.])
                            else:
                                normalized_distr = np.array([distr[0]-diff_from_one/2, distr[1]-diff_from_one/2]) #np.array([distr[0]-abs(diff)/2, distr[1]-abs(diff)/2]) #np.array([0.5-abs(diff), 0.5+abs(diff)])
                    print(f'Before: {distr}\t After:{normalized_distr}')
                    jsd_ispn = compute_jsd(h, normalized_distr)
                    print(f'JSD(GT, iSPN; Intervention = {i}, Variable = {v}) = {jsd_ispn:.4f}   [{h} vs. {normalized_distr}]')
                    jsds.append((jsd_mdn, jsd_made, jsd_ispn))

            # turn of axes
            if dn == "CH":
                axs[a_y, a_x].set_xlim(0,1)
                axs[a_y, a_x].set_ylim(0,0.4)
            else:
                axs[a_y, a_x].set_xlim(-0.5,1.5)
                axs[a_y, a_x].set_ylim(0,1.3)
            #axs[a_y, a_x].set_ylim(bottom=0)
            if plot_with_info:
                axs[a_y, a_x].set_title(f"V:{v},I:{i},D:{dn}", fontsize=5)
            else:
                axs[a_y, a_x].axes.xaxis.set_ticklabels([])
                axs[a_y, a_x].axes.yaxis.set_ticklabels([])

    d_jsds.update({gg: jsds})

for i in range(4):
    m_ispn = []
    m_made = []
    m_mdn = []
    for k in d_jsds:
        m_ispn.append(d_jsds[k][i][2])
        m_made.append(d_jsds[k][i][1])
        m_mdn.append(d_jsds[k][i][0])
    std_ispn = np.round(np.std(m_ispn), decimals=3)
    m_ispn = np.mean(m_ispn)
    std_made = np.round(np.std(m_made), decimals=3)
    m_made = np.mean(m_made)
    std_mdn = np.round(np.std(m_mdn), decimals=3)
    m_mdn = np.mean(m_mdn)
    print(f'On {i} - iSPN {m_ispn:.3f}\pm{str(std_ispn)[1:]}\t MADE {m_made:.3f}\pm{str(std_made)[1:]}\t MDN {m_mdn:.3f}\pm{str(std_mdn)[1:]}')

for k in d_jsds:
    for i in range(4):
        m_ispn = np.round(d_jsds[k][i][2], decimals=3)
        m_made = np.round(d_jsds[k][i][1], decimals=3)
        m_mdn = np.round(d_jsds[k][i][0], decimals=3)
        print(f'On {k}:{i} - iSPN {m_ispn:.3f} & {m_made:.3f} & {m_mdn:.3f}')

if not plot_with_info and plot_compact:
    plt.subplots_adjust(wspace=0, hspace=0)
else:
    plt.tight_layout()
plt.show()
plt.close()
plt.clf()

############################
# validation for plots

# # MADE
# fig, axs = plt.subplots(1,4,figsize=(13,9))
# dn = 'CH'
# i = 'None'
# for ind, v in enumerate(variables[dn]):
#     a_x = ind
#     height = MADE_data[dn][606][i][v][1]
#     pos = normalize(MADE_data[dn][606][i][v][0], a=-0.2, b=1, low=-20, high=100)
#     pos_diff = pos[1] - pos[0]
#     #pos = np.array([pos[0] + i * pos_diff for i in range(len(height))])
#     if dn != "CH":
#         pos += 0.3
#     axs[a_x].bar(x=pos, height=height, color="red", alpha=0.5, width=pos_diff)
#     axs[a_x].set_xlim(-0.2,1)
#     axs[a_x].set_ylim(0.,0.4)
# plt.show()
# plt.close()
# plt.clf()

# # MDN
# fig, axs = plt.subplots(1,4,figsize=(13,9))
# dn = 'CH'
# i = 'None'
# for ind, v in enumerate(variables[dn]):
#     a_x = ind
#     height = MDN_data[dn][606][i][v][1]
#     pos = MDN_data[dn][606][i][v][0]
#     pos_diff = pos[1] - pos[0]
#     #pos = np.array([pos[0] + i * pos_diff for i in range(len(height))])
#     if dn != "CH":
#         pos += 0.3
#     axs[a_x].bar(x=pos, height=height, color="red", alpha=0.5, width=pos_diff)
#     axs[a_x].set_xlim(-0.2,1)
#     axs[a_x].set_ylim(0.,0.4)
# plt.show()
# plt.close()
# plt.clf()

# # GT
# n = 4
# fig, axs = plt.subplots(1,n,figsize=(13,9))
# dn = 'asia'
# i = 'None'
# for ind, v in enumerate(variables[dn][:n]):
#     a_x = ind
#     if v not in gt_dict[dn][str(i)].keys():  # for the CH case
#         v_alt = v[0]
#     else:
#         v_alt = v
#     hist_data = gt_dict[dn][str(i)][v_alt]
#     # weights = np.ones_like(hist_data) / len(hist_data)
#     # gt_hist = axs[a_x].hist(hist_data, label="GT", weights=weights, color='black')
#     h_bins = [0.,0.5,1.]
#     weights = np.ones_like(hist_data) / len(hist_data)
#     h, e = np.histogram(hist_data, bins=h_bins, weights=weights)
#     axs[a_x].bar(range(len(h_bins) - 1), h, width=0.1, edgecolor='k')
#     #axs[a_x].set_xlim(-0.2,1)
#     #axs[a_x].set_ylim(0.,0.4)
# plt.show()
# plt.close()
# plt.clf()