import bnlearn as bn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from glob import glob

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

datasets = ['earthquake']
treatment_dict = {'asia': ['asia','bronc'], 'earthquake': ['Burglary']}
effect_dict = {'asia': ['tub','dysp'], 'earthquake': ['Alarm'] }
for dataset in datasets:
    print(f'Dataset: {dataset} with Sample Size N={N}')
    treatment_nodes = treatment_dict[dataset]
    effect_nodes  = effect_dict[dataset]
    assert (len(treatment_nodes) == len(effect_nodes))
    for ind, int_n in enumerate(treatment_nodes):
        print(f'Effect: {effect_nodes[ind]} || Treatment: {int_n}')

        # ground truth
        model_tr_zero, model_tr_one = get_treatment_models(d[dataset], treatment_node=int_n)
        ate_gt = compute_ATE(model_tr_one, model_tr_zero, effect_node=effect_nodes[ind])
        print(f'Ground Truth ATE: {ate_gt:.4f}')

        # dowhy
        from causalml.inference.meta import LRSRegressor
        y = np.hstack((model_tr_one['data'][effect_nodes[ind]], model_tr_zero['data'][effect_nodes[ind]],))
        treatment = np.hstack((model_tr_one['data'][int_n], model_tr_zero['data'][int_n],))
        X = np.zeros((len(y),1)) #np.vstack((y, treatment)).T # can be anything really
        lr = LRSRegressor()
        te, lb, ub = lr.estimate_ate(X, treatment, y)
        print(f'Dowhy ATE (Linear Regression): {te[0]:.4f}')
