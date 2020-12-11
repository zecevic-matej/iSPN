import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3, linewidth=1000)
import matplotlib.gridspec as gridspec
import SeabornFig2Grid as sfg
import gc

def clean_plt():
    plt.close('all')
    gc.collect()

# '''
# Example with Penguin Data both as Pd Frame and Np Matrix
# '''
# penguins = sns.load_dataset("penguins")
# penguins_np = np.array(penguins)
#
# sns.displot(penguins, x="flipper_length_mm", kind="kde")
# sns.displot(np.array(penguins['flipper_length_mm']), kind='kde')
#
# sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")
# sns.jointplot(x=np.array(penguins_np[:,2],float), y=np.array(penguins_np[:,3],float), hue=np.array(penguins_np[:,0],str), kind="kde")
# plt.show()

'''
create Toy Data, two continuous variables A and F
'''
class SCM_cont():

    def __init__(self, intervene):

        sig_age = 10
        mu_age = 40
        sig_fh = 1
        mu_fh = 10

        age = lambda size: sig_age * np.random.randn(size) + mu_age
        if intervene:
            food_habit = lambda age: sig_fh * np.random.randn(1) + mu_fh
        else:
            food_habit = lambda age: sig_fh * np.random.randn(1) + mu_fh + age

        self.equations = {
            'A': age,
            'F': food_habit,
        }

    def create_data_sample(self, sample_size, as_df=True):

        As = self.equations['A'](sample_size)
        Fs = np.array([self.equations['F'](a)[0] for a in As])

        if as_df:
            return pd.DataFrame.from_dict({'Variable': np.hstack((np.repeat('A',sample_size),np.repeat('F',sample_size))),
                                           'Value': np.hstack((As, Fs))})
        else:
            return {'Age': As, 'Food Habit': Fs}

np.random.seed(0)
N = 10000
as_df = False

scm = SCM_cont(intervene=False)
data_obs = scm.create_data_sample(N, as_df=as_df)
if not as_df:
    print('Observational Data\nFirst 10 samples from a total of {} samples:\n'
          '\tAge         = {}\n'
          '\tFood Habits = {}'.format(N, data_obs['Age'], data_obs['Food Habit']))

scm = SCM_cont(intervene=True)
data_int = scm.create_data_sample(N, as_df=as_df)
if not as_df:
    print('Interventional Data\nFirst 10 samples from a total of {} samples:\n'
          '\tAge         = {}\n'
          '\tFood Habits = {}'.format(N, data_int['Age'], data_int['Food Habit']))

'''
visualizations
'''
#sns.displot(data=data,x='Value', hue='Variable',kind='kde')


clean_plt()
sns.set(style="white", color_codes=True)
fig = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(1, 2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax0.axis('off')
ax1.axis('off')
ax1.set_xlabel('')
ax1.set_ylabel('')
plt.setp([ax0.get_xticklabels(),ax0.get_yticklabels(),ax1.get_xticklabels(),ax1.get_yticklabels()], visible=False)

ax0.set_title('Observational Distribution')
g0 = sns.jointplot(data=pd.DataFrame(data_obs), x='Age', y='Food Habit', xlim = (-10,80), ylim = (-10,80), kind='hex', color='skyblue')
ax1.set_title('Interventional Distribution')
g1 = sns.jointplot(data=pd.DataFrame(data_int), x='Age', y='Food Habit', xlim = (-10,80), ylim = (-10,80), kind='hex', color='green')

mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
fig.suptitle('Effect of Intervention on Joint Distribution')
plt.show()
clean_plt()
