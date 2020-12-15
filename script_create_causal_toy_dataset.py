import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter={'float_kind':"{:.2f}".format})

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

    def __init__(self, discrete=True):

        age = lambda size: np.random.randint(low=18,high=80+1,size=size)
        food_habit = lambda age: 1 if age >= 40 else np.random.binomial(1,0.5)
        if discrete:
            health = lambda age, food_habit: np.random.choice(list(np.arange(1.,4.5,0.5)) + [5.]) #(0 if age >= 60 else (1.5 if age >= 30 else 3)) + food_habit + np.random.binomial(1,0.5)
            mobility = lambda health: 1/2 * health + np.random.binomial(1,0.5)
        else:
            health = lambda age, food_habit: 3/6 * (1 - (age-18)/(80-18)) + 2/6 * food_habit + 1/6 * np.random.binomial(1,0.5)
            mobility = lambda health: 4/5 * health + 1/5 * np.random.binomial(1,0.5)

        self.equations = {
            'A': age,
            'F': food_habit,
            'H': health,
            'M': mobility
        }

    def create_data_sample(self, sample_size):

        As = self.equations['A'](sample_size)
        Fs = np.array([self.equations['F'](a) for a in As])
        Hs = np.array([self.equations['H'](a, Fs[ind]) for ind, a in enumerate(As)])
        Ms = np.array([self.equations['M'](h) for h in Hs])

        return {'A': As, 'F': Fs, 'H': Hs, 'M': Ms}

# create a dataset
np.random.seed(0)
N = 1000
scm = SCM_Health(discrete=False)
data = scm.create_data_sample(N)
print('First 10 samples from a total of {} samples:\n'
      '\tAge         = {}\n'
      '\tFood Habits = {}\n'
      '\tHealth      = {}\n'
      '\tMobility    = {}'.format(N, data['A'][:10], data['F'][:10], data['H'][:10], data['M'][:10]))

# plot a heatmap of Health in sample w.r.t. Age and Food Habits (direct causal parents)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,3))
plt.scatter(data['A'], data['F'], c=data['H'])
plt.colorbar()
plt.title('Health $H$ sampled for {} Persons via SCM'.format(N))
plt.xlabel('Age $A$')
plt.ylabel('Food Habits $F$')
axes = plt.gca()
axes.set_xlim([18-1,80+1])
axes.set_yticks([0,1])
plt.show()

# plot the mean health per age group
mean_health_per_age = []
std_health_per_age = []
for a in range(18,80+1):
    indices = np.where(data['A'] == a)[0]
    corresponding_health_data = [data['H'][ind] for ind in indices]
    mean_health = np.mean(corresponding_health_data)
    std_health = np.std(corresponding_health_data)
    mean_health_per_age.append(mean_health)
    std_health_per_age.append(std_health)

plt.figure(figsize=(12,6))
plt.plot(range(18,80+1), mean_health_per_age)
plt.errorbar(range(18,80+1), mean_health_per_age, yerr=std_health_per_age)
plt.title('Mean Health $H$ per Age group $A$ (Sampled {} Persons via SCM)'.format(N))
plt.xlabel('Age $A$')
plt.ylabel('Health $H$')
axes = plt.gca()
axes.set_xlim([18-1,80+1])
axes.set_ylim([0,1])
plt.show()


'''
discrete Data
'''
# create a dataset
np.random.seed(0)
N = 1000
scm = SCM_Health(discrete=True)
data = scm.create_data_sample(N)
print('(Discrete) First 10 samples from a total of {} samples:\n'
      '\tAge         = {}\n'
      '\tFood Habits = {}\n'
      '\tHealth      = {}\n'
      '\tMobility    = {}'.format(N, data['A'][:10], data['F'][:10], data['H'][:10], data['M'][:10]))

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
plt.title('Health $H$ per Age group $A$ (Sampled {} Persons via SCM)'.format(N))
plt.xlabel('Age $A$')
plt.ylabel('Health $H$')
plt.legend()
axes = plt.gca()
axes.set_xlim([18-1,80+1])
axes.set_ylim([0,5])
plt.show()

save=False

if save:
    import pickle
    with open('causal_health_toy_data_discrete_soft_intervention_Health.pkl', 'wb') as f:
        pickle.dump(data, f)