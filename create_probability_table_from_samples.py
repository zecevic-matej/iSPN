import pandas as pd
import glob
import os
import pickle
import itertools
import numpy as np

p_datasets = 'datasets/data_for_uniform_interventions/'

def load_pickle_data(p):
    with open(p, 'rb') as f:
        data = pickle.load(f)
    return data

p_datasets = glob.glob(os.path.join(p_datasets, "*_100k.pkl")) # big versions of dataset

datasets = [(p.split('intervention_')[1], pd.DataFrame.from_dict(load_pickle_data(p))) for p in p_datasets] # assumes naming convention

pts = {}
for data in datasets:
    interv, df = data
    print('Intervention: {}'.format(interv))
    counts = df.groupby(df.columns.tolist(), as_index=False).size() # validate with  len(np.where((np.array(df) == [80, 1, 2, 1]).all(axis=1))[0])
    probs = counts/counts.sum() # the support is very sparse for our data only 340/12096 states
    print('Support for {} states.'.format(len(probs)))

    np_df = np.array(df)
    domains = [np.unique(np_df[:,i]).tolist() for i in range(np_df.shape[1])]

    states = list(itertools.product(*domains))
    print('Creating probability table for {} states\n'.format(len(states)))

    pt = np.zeros((len(states), len(domains) + 1))
    for ind, s in enumerate(states):
        try:
            p = probs[s]
        except Exception as e:
            p = 0 # no support if not within the calculated frequencies
        pt[ind,:-1] = s
        pt[ind,-1] = p
        print('{}/{}'.format(ind+1, len(states)), end='\r', flush=True)
    assert(pt[:,-1].sum() == 1.)
    df_pt = pd.DataFrame(pt, columns=['A','F','H','M','p'])
    pts.update({interv: df_pt})

save = True
if save:
    p_models = 'models/probability_tables_intervention_data.pkl'
    with open(p_models, 'wb') as f:
        pickle.dump(pts, f)
    print('Saved Probability Tables @ {}'.format(p_models))