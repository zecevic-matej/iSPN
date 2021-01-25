import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter={'float_kind':"{:.2f}".format})
import matplotlib.pyplot as plt
import os
import pickle

"""
Structural Causal Model

    A = N_A, N_A is Uniform distributed, A in N
    F = 1_X(A) OR N_F, N_F is Bernoulli distributed, F in {0,1}
    H = alpha * F + beta * A + gamma * N_H, N_H is Bernoulli distributed and alpha + beta + gamma = 1, H in (0, 1]
    M = delta * H + (1-delta) * N_M, N_M is Bernoulli distributed, M in (0, 1]

    Age -> Food Habit
    Age -> Health
    Food Habit -> Health
    Health -> Mobility
    
"""

class SCM_Health():

    def __init__(self, discrete=True, domains={}):

        if discrete:
            age = lambda size: np.random.randint(low=18,high=80+1,size=size)[0]
            food_habit = lambda age: 1 if age >= 40 else np.random.binomial(1,0.5)
            health = lambda age, food_habit: (0 if age >= 60 else (1.5 if age >= 30 else 3)) + food_habit + np.random.binomial(1,0.5)
            mobility = lambda health: 1/2 * health + np.random.binomial(1,0.5)
        else:
            age = lambda size: np.random.uniform(low=0, high=100, size=size)[0]
            food_habit = lambda age: 0.5 * age + np.random.normal(loc=10,scale=np.sqrt(10))
            health = lambda age, food_habit: 0.008*(100 - age**2) + 0.5 * food_habit + np.random.normal(loc=40,scale=np.sqrt(30))
            mobility = lambda health: 0.5 * health + np.random.normal(loc=20,scale=np.sqrt(10))

        self.equations = {
            'A': age,
            'F': food_habit,
            'H': health,
            'M': mobility
        }

        self.domains = domains
        if domains:
            print("Domains set manually.")
        self.intervention = None
        self.intervention_range = None

        self.discrete = discrete

    def create_data_sample(self, sample_size, domains=True):

        As = np.array([self.equations['A'](1) for _ in range(sample_size)])
        Fs = np.array([self.equations['F'](a) for a in As])
        Hs = np.array([self.equations['H'](a, Fs[ind]) for ind, a in enumerate(As)])
        Ms = np.array([self.equations['M'](h) for h in Hs])

        data = {'A': As, 'F': Fs, 'H': Hs, 'M': Ms}

        if domains and self.discrete:
            domain = lambda N, x: np.unique(data[x]) if N >= 1000 else None
            dA = domain(sample_size, 'A')
            dF = domain(sample_size, 'F')
            dH = domain(sample_size, 'H')
            dM = domain(sample_size, 'M')
            self.domains.update({"A": dA, "F": dF, "H": dH, "M": dM})
            print('*****SCM with Intervention: {} *******\n'
                  '\nDomains / Unique Values for each Variable:\n'
                  '\tAge         = {}\n'
                  '\tFood Habits = {}\n'
                  '\tHealth      = {}\n'
                  '\tMobility    = {}'
                  '\n'.format(self.intervention, dA,dF,dH,dM))

        return data

    def do(self,intervention):
        """
        perform a uniform intervention on a single node
        """

        if intervention is not None and self.discrete:
            if self.domains[intervention] is None:
                print("Please specify the domain!")
                return False
            self.equations[intervention] = lambda *args: np.random.choice(self.domains[intervention])
            print("Performed Uniform Intervention do({}=U({}))".format(intervention,intervention))
            self.intervention = intervention
        elif intervention is not None and not self.discrete:
            low=0
            high=100
            self.equations[intervention] = lambda *args: np.random.uniform(low,high)
            print("Performed Uniform Intervention do({}=U({},{}))".format(intervention,low,high))
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif intervention is None:
            pass

"""
parameters
"""

discrete = False

interventions = [
    (None, "None"),
    ("H","do(H)=U(H)"),
    ("M","do(M)=U(M)"),
    ("A","do(A)=U(A)"),
    ("F","do(F)=U(F)"),
]

domains = {}

np.random.seed(0)
N = 1000

dir_save = "datasets/data_for_uniform_interventions_continuous" # from causal-spn base folder
save = True

"""
simulate SCMs
"""

for interv in interventions:
    interv, interv_desc = interv

    if not discrete: # then continuous

        # create a dataset
        scm = SCM_Health(discrete=False, domains=domains)
        scm.do(interv)
        data = scm.create_data_sample(N, domains=True)
        for ind_d, d in enumerate(['Age','Food Habits','Health','Mobility']):
            e = d[0]
            print('Min {:.2f}\t Max {:.2f}\t Mean {:.2f}\t Median {:.2f}\t STD {:.2f}\t\t - {}'
                  .format(np.min(data[e]), np.max(data[e]), np.mean(data[e]), np.median(data[e]), np.std(data[e]), d))
        print('(Continuous) First 10 samples from a total of {} samples:\n'
              '\tAge         = {}\n'
              '\tFood Habits = {}\n'
              '\tHealth      = {}\n'
              '\tMobility    = {}'
              '\n\n***********************************\n\n'.format(N,
                                                               data['A'][:10],
                                                               data['F'][:10],
                                                               data['H'][:10],
                                                               data['M'][:10]))

        # plot the median health per age group
        plt.figure(figsize=(12,7))
        for v in ['Food Habits','Health','Mobility']:
            median_var_per_age = []
            mean_var_per_age = []
            std_var_per_age = []
            age_intervals = [(0, 10), (10, 30), (30, 55), (55, 75), (75, 100)]
            for a in age_intervals:
                indices = np.where(np.logical_and(data['A'] > a[0],data['A'] < a[1]))[0]
                corresponding_var_data = [data[v[0]][i] for i in indices]
                median_var = np.median(corresponding_var_data)
                mean_var = np.mean(corresponding_var_data)
                std_var = np.std(corresponding_var_data)
                median_var_per_age.append(median_var)
                mean_var_per_age.append(mean_var)
                std_var_per_age.append(std_var)

            #plt.plot(range(len(age_intervals)), median_var_per_age, label='Median',)
            e = v[0]
            p = plt.plot(range(len(age_intervals)), mean_var_per_age, label='{} |All Data {:.1f}, {:.1f}, {:.1f}|'.format(v,np.mean(data[e]), np.min(data[e]), np.max(data[e])))
            plt.errorbar(range(len(age_intervals)), mean_var_per_age, yerr=std_var_per_age, color=p[0].get_color())
            plt.title('Intervention: {} {}\nContinuous Data Mean Values per Age intervals x<a<y (Sampled {} Persons via SCM)\nVariable Name |All Data Mean, Min, Max|'.format(interv_desc, scm.intervention_range,N))
            plt.xlabel('Age $A$')
            plt.ylabel('Mean for Variable in Interval')
            plt.xticks(range(len(age_intervals)), [str(x) for x in age_intervals])
        plt.ylim(-10,70)
        plt.legend(bbox_to_anchor=[0.5, -0.11], loc='center', ncol=3)
        axes = plt.gca()
        plt.show()

    else:
        '''
        discrete Data
        '''
        # create a dataset
        scm = SCM_Health(discrete=True, domains=domains)
        scm.do(interv)
        data = scm.create_data_sample(N, domains=True)
        if scm.domains is not None and interv is None:
            domains = scm.domains
            print("Gathered observational domains for variables.")
        # print("\nMedian Values for (A,F,H,M) = ({},{},{},{})\n".format(
        #     np.median(data["A"]),np.median(data["F"]),np.median(data["H"]),np.median(data["M"])
        # ))
        print('(Discrete) First 10 samples from a total of {} samples:\n'
              '\tAge         = {}\n'
              '\tFood Habits = {}\n'
              '\tHealth      = {}\n'
              '\tMobility    = {}'
              '\n\n***********************************\n\n'.format(N,
                                                               data['A'][:10],
                                                               data['F'][:10],
                                                               data['H'][:10],
                                                               data['M'][:10]))

        # plot the median health per age group
        median_health_per_age = []
        mean_health_per_age = []
        std_health_per_age = []
        for a in range(18,80+1):
            indices = np.where(data['A'] == a)[0]
            corresponding_health_data = [data['H'][ind] for ind in indices]
            median_health = np.median(corresponding_health_data)
            mean_health = np.mean(corresponding_health_data)
            std_health = np.std(corresponding_health_data)
            median_health_per_age.append(median_health)
            mean_health_per_age.append(mean_health)
            std_health_per_age.append(std_health)

        plt.figure(figsize=(12,6))
        plt.plot(range(18,80+1), median_health_per_age, label='Median', color='b')
        plt.plot(range(18,80+1), mean_health_per_age, label='Mean', color='green')
        plt.errorbar(range(18,80+1), mean_health_per_age, yerr=std_health_per_age, color='green')
        plt.title('Intervention: {}\nHealth $H$ per Age group $A$ (Sampled {} Persons via SCM)'.format(interv,N))
        plt.xlabel('Age $A$')
        plt.ylabel('Health $H$')
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([18-1,80+1])
        axes.set_ylim([0,5])
        plt.show()

    if save:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
            print("Created directory: {}".format(dir_save))
        type = 'discrete' if discrete else 'continuous'
        save_location = os.path.join(dir_save,
                               'causal_health_toy_data_{}_intervention_{}_N{}.pkl'.format(type, interv_desc,N))
        with open(save_location,'wb') as f:
            pickle.dump(data, f)
            print("Saved Data @ {}".format(save_location))