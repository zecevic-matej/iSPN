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
        """
        pre-computed models are stored as csv probability tables which are loaded with this
        """
        return pd.read_csv(model_spn_path)

    def get_state_space(self, data, verbose=False):
        """
        create all state space combinations
        given the discrete space, this is then used to compute probability tables
        index_encoding refers to assigning 'index classes' to the actual values (e.g. 0.5, 1, 1.5 == 0,1,2)
        """
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

        self.states_indices = {'A': 0, 'F': 1, 'H': 2, 'M': 3}

        return states

    def get_conditional_probability_table_entry(self, cpt, y, x):
        """
        if provided a CSPN-style Conditional Probability Table (cpt)
        then simply return the correct entry
        """
        return cpt.iloc[y,x+1]

    def get_conditional_probability_via_joint(self, pt, Y, X):
        """
        if provided a SPN-style joint model als Probability Table (pt)
        then using then create the Conditional PT for Y given X
        by using Bayes Rule i.e., p(Y,X)/p(X) = p(Y|X)
        """
        state_indices = [self.states_indices[k] for k in (Y,X)]
        states_yx = np.unique(np.array(self.states)[:, state_indices], axis=0)
        assert(states_yx.shape[1] == 2)
        states_y = np.unique(states_yx[:,0])
        states_x = np.unique(states_yx[:,1])

        probs = []
        visited_states = []
        for ix, x in enumerate(states_x):
            p_x = np.sum([pt['p'][i] for i in np.where(pt[X] == x)[0]])
            if np.allclose(p_x, 0):
                print('Encountered Division by Zero.')
                import pdb; pdb.set_trace()
            for iy, y in enumerate(states_y):
                p_yx = np.sum([pt['p'][i] for i in np.where((pt[[Y, X]] == (y, x)).all(axis=1))[0]])
                p_y_given_x = p_yx / p_x
                #print('p(Y={},X={}) = {}, p(x) = {} -> p(y|x) = {}'.format(y, x, p_yx, p_x, p_y_given_x))
                probs.append(p_y_given_x)
                visited_states.append([y,x])
                print('     {}/{}           '.format((ix*len(states_y) + iy) + 1, len(states_yx)), end='\r', flush=True)
        cpt = pd.DataFrame(np.hstack((np.array(visited_states), np.array(probs)[:, np.newaxis])),
                          columns=[Y, X, 'p'])
        k = 'p({}|{}) (SPN)'.format(Y,X)
        self.approximations.update({k: cpt})
        print('Pre-computed the probability table for {} using the existing SPN joint model.'.format(k))

    def p(self, query):
        """
        use the registered models and approximations to compute a given query
        using the current Causal SPN approach
        """

        if query == 'p(A,F,M|do(H=U(H))':
            # truncated factorization
            # assumes usage of conditional probabilities from CSPN

            if query in causal_spn.approximations.keys():
                print('Already pre-computed.')
                return

            probs = []
            for i, s in enumerate(self.states):
                a,f,h,m = s
                p_a = np.sum([causal_spn.models['p(A,F,H,M)'].iloc[x]['p'] for x in np.where(causal_spn.models['p(A,F,H,M)']['A'] == a)[0]])
                p_f_given_a = self.get_conditional_probability_table_entry(causal_spn.models['p(F|A)'], f, a)
                p_m_given_h = self.get_conditional_probability_table_entry(causal_spn.models['p(M|H)'], m, h)
                p_h_given_do_h = 0.125 # U(H) is uniform on 8-dimensional H space, only necessary because of non-scalar intervention
                prob = p_a * p_f_given_a * p_h_given_do_h * p_m_given_h
                probs.append(prob)
                print('     {}/{}           '.format(i + 1, len(self.states)), end='\r', flush=True)

            pt = pd.DataFrame(np.hstack((np.array(self.states), np.array(probs)[:, np.newaxis])), columns=['A', 'F', 'H', 'M', 'p'])
            self.approximations.update({'p(A,F,M|do(H=U(H))': pt})

        elif query == 'p(A,F,H,M) (SPN)':
            # regular bayesian net factorization
            # assumes that the conditional probabilities are coming from the joint estimate of the SPN

            if query in causal_spn.approximations.keys():
                print('Already pre-computed.')
                return

            probs = []
            for i, s in enumerate(self.states):
                a,f,h,m = s
                p_a = np.sum([causal_spn.models['p(A,F,H,M)'].iloc[x]['p'] for x in np.where(causal_spn.models['p(A,F,H,M)']['A'] == a)[0]])
                p_f_given_a = np.sum([causal_spn.approximations['p(F|A) (SPN)'].iloc[x]['p'] for x in np.where(causal_spn.approximations['p(F|A) (SPN)']['A'] == a)[0]])
                p_m_given_h = np.sum([causal_spn.approximations['p(M|H) (SPN)'].iloc[x]['p'] for x in np.where(causal_spn.approximations['p(M|H) (SPN)']['A'] == a)[0]])
                p_h_given_f_and_a = ...
                prob = p_a * p_f_given_a * p_h_given_f_and_a * p_m_given_h
                probs.append(prob)
                print('     {}/{}           '.format(i + 1, len(self.states)), end='\r', flush=True)

            pt = pd.DataFrame(np.hstack((np.array(self.states), np.array(probs)[:, np.newaxis])), columns=['A', 'F', 'H', 'M', 'p'])
            self.approximations.update({'p(A,F,H,M) (SPN)': pt})


models = {
    'p(A,F,H,M)': 'models/probability_table_spn_causal_toy_dataset.pkl',
    'p(F|A)': 'models/probability_table_cspn_causal_toy_dataset_p(F|A).pkl',
    'p(M|H)': 'models/probability_table_cspn_causal_toy_dataset_p(M|H).pkl',
    'p(A,F,M|do(H=U(H))': 'models/probability_table_spn_causal_toy_dataset_soft_intervention_Health.pkl'
}
data_obs = load_data('datasets/causal_health_toy_data_discrete.pkl')

causal_spn = CausalSPN(models=models, data_obs=data_obs)

jsd_obs_vs_obs = jensenshannon(causal_spn.models['p(A,F,H,M)']['p'], causal_spn.models['p(A,F,H,M)']['p'], base=2)
print('Jensen-Shannon-Divergence Obs vs. Obs : {}\n'.format(jsd_obs_vs_obs))


jsd_obs_vs_int = jensenshannon(causal_spn.models['p(A,F,H,M)']['p'], causal_spn.models['p(A,F,M|do(H=U(H))']['p'], base=2)
print('Jensen-Shannon-Divergence Obs vs. Int : {}\n'.format(jsd_obs_vs_int))

query = 'p(A,F,M|do(H=U(H))'
causal_spn.p(query=query)
jsd_causalspn_vs_int = jensenshannon(causal_spn.approximations[query]['p'], causal_spn.models['p(A,F,M|do(H=U(H))']['p'], base=2)
print('Jensen-Shannon-Divergence Causal SPN vs. Int : {}'.format(jsd_causalspn_vs_int))

causal_spn.get_conditional_probability_via_joint(pt=causal_spn.models['p(A,F,H,M)'], Y='F', X='A')
causal_spn.get_conditional_probability_via_joint(pt=causal_spn.models['p(A,F,H,M)'], Y='M', X='H')
...

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
