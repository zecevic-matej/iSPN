from spn.io.Text import str_to_spn
from spn.algorithms.Inference import likelihood
import numpy as np

def load_spn_model(model_path):
    with open(model_path, 'r') as f:
        str_spn = f.readlines()
    str_spn = ''.join(str_spn)
    return str_to_spn(str_spn)

'''
predicting with example model (continuous)
'''
model = load_spn_model('models/spn_continuous.txt')

# p(F=1), A,F,H,M
query = np.array([np.nan, 1, np.nan, np.nan])[np.newaxis,:]
likelihood(model, query)

# p(A=a)
likelihoods = []
for a in np.arange(18,80):
    query = np.array([a, np.nan, np.nan, np.nan])[np.newaxis,:]
    likelihoods.append(likelihood(model, query))

# p(A=a,F=1) is it this?? (certainly not, p(F|A) but that you can get by dividing throuhg p(A))
likelihoods = []
for a in np.arange(18,80):
    query = np.array([a, 1, np.nan, np.nan])[np.newaxis,:]
    likelihoods.append(likelihood(model, query))

# l(H=0.7)
query = np.array([np.nan, np.nan, 0.7, np.nan])[np.newaxis,:]
likelihood(model,query)

'''
create probability table for model
'''
import itertools
import pickle
import pandas as pd

model_path = 'models/spn_discrete_soft_intervention_Health.txt'
model = load_spn_model(model_path)
print('Loaded Model {}'.format(model_path))

path_data = 'datasets/causal_health_toy_data_discrete_soft_intervention_Health.pkl'
with open(path_data, "rb") as f:
    data = pickle.load(f)

states_A = np.unique(data['A'])
states_F = np.unique(data['F'])
states_H = np.unique(data['H'])
states_M = np.unique(data['M'])
print('State Spaces: \nA={}\nF={}\nH={}\nM={}'.format(states_A,states_F,states_H,states_M))

do_index_encoding=True
if do_index_encoding:
    states_A = np.arange(len(states_A))
    states_F = np.arange(len(states_F))
    states_H = np.arange(len(states_H))
    states_M = np.arange(len(states_M))
    print('Index Encoded State Spaces: \nA={}\nF={}\nH={}\nM={}'.format(states_A, states_F, states_H, states_M))

print('\nPlease make sure your Model was trained on this Dataset!')

states = list(itertools.product(states_A, states_F, states_H, states_M))
print('Creating probability table for {} states'.format(len(states)))

probs = []
for i, s in enumerate(states):
    probs.append(likelihood(model,np.array(s)[np.newaxis,:])[0][0])
    print('{}/{}           '.format(i+1, len(states)), end='\r', flush=True)

pt = pd.DataFrame(np.hstack((np.array(states), np.array(probs)[:,np.newaxis])), columns=['A','F','H','M','p'])
pt.to_csv('models/probability_table_spn_causal_toy_dataset_soft_intervention_Health.pkl')


'''
compute JSD
'''
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pandas as pd

pt_obs = pd.read_csv('models/probability_table_spn_causal_toy_dataset.pkl')
pt_int = pd.read_csv('models/probability_table_spn_causal_toy_dataset_soft_intervention_Health.pkl')

# entropy(p,q,base=2) # relative entropy == kl divergence, default log is to base e, np.sum(kl_div(p,q))

jsd = jensenshannon(pt_obs['p'], pt_int['p'], base=2)