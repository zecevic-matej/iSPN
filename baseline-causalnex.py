# silence warnings
import warnings
warnings.filterwarnings("ignore")
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import pandas as pd
import numpy as np
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
import pickle
import os
import matplotlib.pyplot as plt

class BnLearnDataset():
    def __init__(self, bif, whitelist):
        p = './datasets/other/benchmark_data_for_uniform_interventions/{}_uniform_interventions_N10000.pkl'.format(bif)
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
        self.xs_all = np.vstack(xs_all)[random_sorting]
        self.ys_all = np.vstack(ys_all)[random_sorting]

np.random.seed(0)
save_dir = None#'causalnex_CBN_results_asia'

whitelist = [None, "lung"] # all interventions to be considered, if None then everything is considered
refit = False # when True fit separate CBNs, not one on which we intervene upon

bnl_d = BnLearnDataset('asia', whitelist=whitelist)

sm = StructureModel()
sm.add_edges_from(list(bnl_d.data[0]["model"].edges)) # from observational model
viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    viz.draw(os.path.join(save_dir, "causalnex_asia_observational.png"))
bn = BayesianNetwork(sm)

colors = ['pink', 'blue', 'orange', 'lightgreen', 'yellow', 'red', 'cyan', 'purple'] # ASSUMES: NEVER more than 8 variables
fig_len_x = 2
fig_len_y = 4
list_vars_per_interv = [bnl_d.data[i]['data'].columns.tolist() for i in range(len(bnl_d.intervention))]
low = -0.25
high = 1.25
last_intervention = None
marginals_per_interv = {}
for ind, interv_desc in enumerate(bnl_d.intervention):
    fig, axs = plt.subplots(fig_len_x, fig_len_y, figsize=(12, 10))
    comment = ''

    data = bnl_d.data[ind]['data']

    if interv_desc is not None:
        if refit:
            sm = StructureModel()
            sm.add_edges_from(list(bnl_d.data[ind]["model"].edges))  # from observational model
            viz = plot_structure(
                sm,
                graph_attributes={"scale": "0.5"},
                all_node_attributes=NODE_STYLE.WEAK,
                all_edge_attributes=EDGE_STYLE.WEAK)
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                viz.draw(os.path.join(save_dir, "causalnex_asia_refitted_interv_{}.png".format(interv_desc)))
            sm = sm.get_largest_subgraph() # can only ever cover one graph
            bn = BayesianNetwork(sm)
            bn = bn.fit_node_states(data)
            bn = bn.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")
            ie = InferenceEngine(bn)
            print("Fitted New Causal Bayesian Network to Intervention on {}".format(interv_desc))
        else:
            if last_intervention is not None:
                ie.reset_do(interv_desc)
            assert(interv_desc in bn.nodes)
            ie.do_intervention(interv_desc,{0: 0.5, 1: 0.5}) # assumes uniform interventions on binary variables
            print("Do-Calculus: Performed Uniform Intervention on {}".format(interv_desc))
            last_intervention = interv_desc
    else:
        bn = bn.fit_node_states(data)
        bn = bn.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")
        ie = InferenceEngine(bn)
        print("Fitted Causal Bayesian Network")

    marginals = ie.query()
    if interv_desc is not None and refit:
        # because they don't implement probabilities for separate, without-parents nodes
        for n in bn.node_states.keys():
            if n not in list(marginals.keys()):
                marginals.update({n: {0: 0, 1: 0}})
    marginals_per_interv.update({interv_desc: marginals})

    variables = list_vars_per_interv[0] # assumes that all datasets have the same variables, makes sure that ordering is consistent
    for ind_d, d in enumerate(variables):

        # gt distribution
        hist_data = data[d]
        weights = np.ones_like(hist_data)/len(hist_data)
        h = axs.flatten()[ind_d].hist(hist_data, color=colors[ind_d], label="GT", weights=weights, edgecolor=colors[ind_d], alpha=1)
        axs.flatten()[ind_d].set_title('{}'.format(d))
        axs.flatten()[ind_d].set_xlim(low, high)

        # cbn
        axs.flatten()[ind_d].bar(x=[0,1],height=[marginals[d][0],marginals[d][1]],width=0.5,label="CBN",alpha=0.6)

        axs.flatten()[ind_d].legend(prop={'size':9})
        axs.flatten()[ind_d].set_ylim(0,1)
        print("Variable: {}        ".format(d), end="\r", flush=True)
    plt.suptitle('Intervention: {}'.format(interv_desc))
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if refit:
            prefix = "_refitted"
        else:
            prefix = ""
        save_loc = os.path.join(save_dir, "causalnex_CBN{}_int_{}.png".format(prefix,interv_desc))
        plt.savefig(save_loc)
        print('Saved @ {}'.format(save_loc))
    else:
        plt.show()