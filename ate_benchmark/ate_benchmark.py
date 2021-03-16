import bnlearn as bn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from glob import glob
import pandas as pd

N=100000
l_data_paths = glob(f"./data/*_N{N}.pkl")

d = {}
for p in l_data_paths:
    with open(p, 'rb') as f:
        data = pickle.load(f)
    d.update({os.path.basename(p).split("_")[0]: data})

def print_cpds(model):
    cpts = model['model'].get_cpds()
    for cpt in cpts:
        var = cpt.variable
        conditions = cpt.variables.copy()
        conditions.remove(var)
        if conditions:
            print('\np({}|{})\n{}\n'.format(var, ','.join(conditions), cpt))
        else:
            print('\np({})\n{}\n'.format(var, cpt))

def get_treatment_models(list_models, treatment_node):
    for ind, m in enumerate(list_models):
        if m['interv'] == treatment_node and m['interv_type'] == 'zero' and list_models[ind+1]['interv_type'] == 'one':
            # assumes model ordering like: zero tr, one tr, ...
            return m, list_models[ind+1]

def compute_expectation(pt):
    # for binary treatment and effect this is just p(e=1)
    return pt[1]

def compute_sample_mean(samples):
    # for high number of samples this converges to expected value
    # https://math.stackexchange.com/questions/715629/proving-a-sample-mean-converges-in-probability-to-the-true-mean
    return 1/len(samples) * sum(samples)

def compute_ATE(model_tr_one, model_tr_zero, effect_node):
    #cpt = model_tr_one
    e_tr_one = compute_sample_mean(model_tr_one['data'][effect_node]) #compute_expectation(cpt[:,1])
    e_tr_zero = compute_sample_mean(model_tr_zero['data'][effect_node]) #compute_expectation(cpt[:,0])
    return e_tr_one - e_tr_zero

def create_graph_from_adj(adj):
    import networkx
    # import pydot
    nxgraph = networkx.from_pandas_adjacency(adj, create_using=networkx.DiGraph)
    graph = networkx.drawing.nx_pydot.to_pydot(nxgraph)
    dot_string = graph.to_string().replace('\n', '')
    return graph, dot_string

datasets = ['asia','earthquake']
treatment_dict = {'asia': ['asia','bronc'], 'earthquake': ['Burglary']}
effect_dict = {'asia': ['tub','dysp'], 'earthquake': ['Alarm'] }
for dataset in datasets:
    model_obs = d[dataset][0]

    print(f'\n++++\nDataset: {dataset} with Sample Size N={N}')
    treatment_nodes = treatment_dict[dataset]
    effect_nodes  = effect_dict[dataset]
    assert (len(treatment_nodes) == len(effect_nodes))
    for ind, int_n in enumerate(treatment_nodes):
        print(f'\nEffect: {effect_nodes[ind]} || Treatment: {int_n}')

        # ground truth
        model_tr_zero, model_tr_one = get_treatment_models(d[dataset], treatment_node=int_n)
        ate_gt = compute_ATE(model_tr_one, model_tr_zero, effect_node=effect_nodes[ind])
        print(f'Ground Truth ATE: {ate_gt:.4f}')

        # GT conditional (should be identical to ATE if no confounding)
        model_cond_one = model_obs['data'][effect_nodes[ind]][np.where(model_obs['data'][int_n] == 1)[0]]
        model_cond_zero = model_obs['data'][effect_nodes[ind]][np.where(model_obs['data'][int_n] == 0)[0]]
        conditional_diff = compute_sample_mean(model_cond_one) - compute_sample_mean(model_cond_zero)
        print(f'Ground Truth Conditional Diff: {conditional_diff:.4f} (should be same as ATE if no confounding)')

        # causalml
        from causalml.inference.meta import LRSRegressor
        y = np.hstack((model_tr_one['data'][effect_nodes[ind]], model_tr_zero['data'][effect_nodes[ind]],))
        treatment = np.hstack((model_tr_one['data'][int_n], model_tr_zero['data'][int_n],))
        X = np.zeros((len(y),1)) #np.vstack((y, treatment)).T # can be anything really
        lr = LRSRegressor()
        te, lb, ub = lr.estimate_ate(X, treatment, y)
        print(f'CausalML ATE (Linear Regression): {te[0]:.4f}')

        # dowhy
        from dowhy import CausalModel
        graph, dot_string = create_graph_from_adj(model_obs['adjmat'])
        # visualize using graph.write_png('name.png')
        df = pd.concat([model_tr_one['data'], model_tr_zero['data']], ignore_index=True)
        model = CausalModel(
            data=df,
            treatment=int_n,
            outcome=effect_nodes[ind],
            graph=dot_string)
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand,method_name='backdoor.linear_regression')#"backdoor.propensity_score_matching")
        print(f'Dowhy ATE (Linear Regression): {estimate.value:.4f}')

# creating Kidney Stone example data
D = pd.DataFrame(np.zeros((700, 3)),columns=['T','Z','R'])
A = pd.DataFrame(np.array([[False, False, True],[True, False, True],[False, False, False]]), columns=['T','Z','R'],index=['T','Z','R'])
D.loc[:562-1,'R'] = 1
#D.loc[273:562+61-1,'T'] = 0
D.loc[:273-1,'T'] = 1
D.loc[562+61:,'T'] = 1
D.loc[81:273-1,'Z'] = 1
D.loc[562+61+6:,'Z'] = 1
D.loc[273+234:562-1,'Z'] = 1
D.loc[562+36:562+61-1,'Z'] = 1
assert(len(D.loc[(D['R']==1) & (D['T']==1)]) == 273)
assert(len(D.loc[(D['R']==1) & (D['T']==0)]) == 289)
assert(len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==0)]) == 81)
assert(len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==1)]) == 192)
assert(len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==0)]) == 234)
assert(len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==1)]) == 55)
print(f'\n++++\nKidney Stone (Simpson Paradox) Toy Dataset with Sample Size N={len(D)}')

pr_R1_g_TA_Z0 = len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==0)])/(len(D.loc[(D['R']==0) & (D['T']==1) & (D['Z']==0)]) + len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==0)]))
pr_R1_g_TA_Z1 = len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==1)])/(len(D.loc[(D['R']==0) & (D['T']==1) & (D['Z']==1)]) + len(D.loc[(D['R']==1) & (D['T']==1) & (D['Z']==1)]))
pr_R1_g_TB_Z0 = len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==0)])/(len(D.loc[(D['R']==0) & (D['T']==0) & (D['Z']==0)]) + len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==0)]))
pr_R1_g_TB_Z1 = len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==1)])/(len(D.loc[(D['R']==0) & (D['T']==0) & (D['Z']==1)]) + len(D.loc[(D['R']==1) & (D['T']==0) & (D['Z']==1)]))
pr_Z0 = len(D.loc[(D['Z']==0)]) / len(D)
pr_Z1 = len(D.loc[(D['Z']==1)]) / len(D)
tr_A = pr_R1_g_TA_Z0 * pr_Z0 + pr_R1_g_TA_Z1 * pr_Z1
tr_B = pr_R1_g_TB_Z0 * pr_Z0 + pr_R1_g_TB_Z1 * pr_Z1
ate_gt = tr_A - tr_B
print(f'Ground Truth ATE (NO interv data just via do-calculus): {ate_gt:.4f}')

tr_A = len(D.loc[(D['R']==1) & (D['T']==1)])/len(D.loc[(D['T']==1)])
tr_B = len(D.loc[(D['R']==1) & (D['T']==0)])/len(D.loc[(D['T']==0)])
conditional_diff = tr_A - tr_B
print(f'Ground Truth Conditional Diff: {conditional_diff:.4f} (should be same as ATE if no confounding)')

# when using this then the solution settles onto the conditional #X = np.zeros((len(D), 1))  # np.vstack((y, treatment)).T # can be anything really
X = np.random.randn(len(D), 1) #D
lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, D['T'], D['R'])
print(f'CausalML ATE (Linear Regression): {te[0]:.4f}')

graph, dot_string = create_graph_from_adj(A)
model = CausalModel(
    data=D,
    treatment='T',
    outcome='R',
    graph=dot_string)
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand,
                                 method_name='backdoor.linear_regression')  # "backdoor.propensity_score_matching")
print(f'Dowhy ATE (Linear Regression): {estimate.value:.4f}')