import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import itertools
import pickle

def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

class CausalSPN():

    def __init__(self, models, data_obs):

        self.models = {}
        for i, k in enumerate(models):
            self.models.update({k: self.load_probabilty_table(models[k])})
        self.states = self.get_state_space(data_obs)
        self.approximations = {}

    def load_probabilty_table(self, model_spn_path):
        return pd.read_csv(model_spn_path)

    def get_state_space(self, data, verbose=False):
        states_A = np.unique(data['A'])
        states_F = np.unique(data['F'])
        states_H = np.unique(data['H'])
        states_M = np.unique(data['M'])
        if verbose:
            print('State Spaces: \nA={}\nF={}\nH={}\nM={}'.format(states_A, states_F, states_H, states_M))

        do_index_encoding = True
        if do_index_encoding:
            states_A = np.arange(len(states_A))
            states_F = np.arange(len(states_F))
            states_H = np.arange(len(states_H))
            states_M = np.arange(len(states_M))
            print('Using Index Encoding.')
            if verbose:
                print('Index Encoded State Spaces: \nA={}\nF={}\nH={}\nM={}'.format(states_A, states_F, states_H, states_M))

        print('\nPlease make sure your models were trained using this Dataset!\n')

        states = list(itertools.product(states_A, states_F, states_H, states_M))
        print('|S| = {}'.format(len(states)))
        return states

    def get_conditional_probability_table_entry(self, cpt, y, x):
        return cpt.iloc[y,x+1]

    def p(self, query):

        if query == 'p(A,F,M|do(H=U(H))':

            if query in causal_spn.approximations.keys():
                print('Already pre-computed.')
                return

            probs = []
            for i, s in enumerate(self.states):
                a,f,h,m = s
                p_a = np.sum([causal_spn.models['p(A,F,M,H)'].iloc[x]['p'] for x in np.where(causal_spn.models['p(A,F,M,H)']['A'] == a)[0]])
                p_f_given_a = self.get_conditional_probability_table_entry(causal_spn.models['p(F|A)'], f, a)
                p_m_given_h = self.get_conditional_probability_table_entry(causal_spn.models['p(M|H)'], m, h)
                p_h_given_do_h = 0.125 # U(H) is uniform on 8-dimensional H space, only necessary because of non-scalar intervention
                prob = p_a * p_f_given_a * p_h_given_do_h * p_m_given_h
                probs.append(prob)
                print('     {}/{}           '.format(i + 1, len(self.states)), end='\r', flush=True)

            pt = pd.DataFrame(np.hstack((np.array(self.states), np.array(probs)[:, np.newaxis])), columns=['A', 'F', 'H', 'M', 'p'])
            self.approximations.update({'p(A,F,M|do(H=U(H))': pt})

        elif query == 'p(A,F,M,H)':

            if query in causal_spn.approximations.keys():
                print('Already pre-computed.')
                return

            probs = []
            for i, s in enumerate(self.states):
                a,f,h,m = s
                p_a = np.sum([causal_spn.models['p(A,F,M,H)'].iloc[x]['p'] for x in np.where(causal_spn.models['p(A,F,M,H)']['A'] == a)[0]])
                p_f_given_a = self.get_conditional_probability_table_entry(causal_spn.models['p(F|A)'], f, a)
                p_m_given_h = self.get_conditional_probability_table_entry(causal_spn.models['p(M|H)'], m, h)
                p_h_given_f_and_a = ...
                prob = p_a * p_f_given_a * p_h_given_f_and_a * p_m_given_h
                probs.append(prob)
                print('     {}/{}           '.format(i + 1, len(self.states)), end='\r', flush=True)

            pt = pd.DataFrame(np.hstack((np.array(self.states), np.array(probs)[:, np.newaxis])), columns=['A', 'F', 'H', 'M', 'p'])
            self.approximations.update({'p(A,F,M|do(H=U(H))': pt})


models = {
    'p(A,F,M,H)': 'models/probability_table_spn_causal_toy_dataset.pkl',
    'p(F|A)': 'models/probability_table_cspn_causal_toy_dataset_p(F|A).pkl',
    'p(M|H)': 'models/probability_table_cspn_causal_toy_dataset_p(M|H).pkl',
    'p(A,F,M|do(H=U(H))': 'models/probability_table_spn_causal_toy_dataset_soft_intervention_Health.pkl'
}
data_obs = load_data('datasets/causal_health_toy_data_discrete.pkl')

causal_spn = CausalSPN(models=models, data_obs=data_obs)

jsd_obs_vs_obs = jensenshannon(causal_spn.models['p(A,F,M,H)']['p'], causal_spn.models['p(A,F,M,H)']['p'], base=2)
print('Jensen-Shannon-Divergence Obs vs. Obs : {}\n'.format(jsd_obs_vs_obs))


jsd_obs_vs_int = jensenshannon(causal_spn.models['p(A,F,M,H)']['p'], causal_spn.models['p(A,F,M|do(H=U(H))']['p'], base=2)
print('Jensen-Shannon-Divergence Obs vs. Int : {}\n'.format(jsd_obs_vs_int))

query = 'p(A,F,M|do(H=U(H))'
causal_spn.p(query=query)
jsd_causalspn_vs_int = jensenshannon(causal_spn.approximations[query]['p'], causal_spn.models['p(A,F,M|do(H=U(H))']['p'], base=2)
print('Jensen-Shannon-Divergence Causal SPN vs. Int : {}'.format(jsd_causalspn_vs_int))


'''
visualize
'''
import matplotlib.pyplot as plt
height = [jsd_obs_vs_obs, jsd_obs_vs_int, jsd_causalspn_vs_int, 100.]
bars = ('Obs vs Obs', 'Obs vs Int', 'Causal SPN (cspn based)\nvs Int', 'Obs Approx. (cspn based)\nvs Obs')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=['blue', 'blue', 'red', 'black'])
plt.xticks(y_pos, bars)
plt.ylim(0,1)
plt.title('Jensen-Shannon-Divergence across all {} probabilities/states'.format(len(causal_spn.states)))
plt.show()
