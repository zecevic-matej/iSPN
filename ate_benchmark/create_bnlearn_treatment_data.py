import bnlearn as bn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# general parameters
# check out https://www.bnlearn.com/bnrepository/discrete-small.html#asia
bif_file = 'asia.bif' # available are: earthquake, cancer, sachs, survey
visualize_immediate = False
visualize_final = True
N = 100000
interventions = ["asia",'bronc']
save_dir = './data/'

# load model
model_obs = bn.import_DAG(bif_file)
model_obs.update({'interv': None})

# visualize_immediate graph
if visualize_immediate:
    bn.plot(model_obs)

# plot CPTs
#bn.print_CPD(model)
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
print_cpds(model_obs)

# generate samples
df_obs = bn.sampling(model_obs, n=N)
model_obs.update({'data': df_obs})

# example intervention on tub
def intervene(model, var, intervention='uniform'):
    cpts = model['model'].get_cpds()
    modeled_vars = [c.variable for c in cpts]
    assert(len(modeled_vars) == len(cpts))
    # find cpt of node to be intervened one
    sel_cpt = cpts[modeled_vars.index(var)]
    conditions = sel_cpt.variables.copy()
    conditions.remove(var)
    # overwrite cpt with intervention cpt
    if intervention == 'uniform':
        sel_cpt.values = np.array([0.5, 0.5])
    elif intervention == 'one':
        sel_cpt.values = np.array([0., 1.])
    elif intervention == 'zero':
        sel_cpt.values = np.array([1., 0.])
    else:
        raise Exception('No Intervention possible.')
    # remove parents edges (just within cpt)
    sel_cpt.variables = [var]
    sel_cpt.cardinality = [len(sel_cpt.values),1]
    model['model'] = model['model'].do(var)
    model['adjmat'][var] = False # columns in adjmat are targets i.e., in-going arrows, which are deleted
    print('\nIntervened on {}'.format(var))
    print_cpds(model)

models = [model_obs]
for interv in interventions:
    for interv_type in ['zero', 'one']:
        models.append({'model': model_obs['model'].copy(), 'adjmat': model_obs['adjmat'].copy(), 'interv': interv, 'interv_type': interv_type})
        intervene(models[-1], interv, intervention=interv_type)
        if visualize_immediate:
            bn.plot(models[-1])
        models[-1].update({'data': bn.sampling(models[-1], n=N)})

if visualize_final:
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10,8))
    for i, m in enumerate(models):
        sns.heatmap(models[i]['adjmat'], vmin=0, vmax=1, ax=axs.flatten()[i])
        axs.flatten()[i].set_title('Intervention on {}'.format(models[i]['interv']))
    plt.suptitle('Graphs for {}'.format(bif_file))
    plt.show()

if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loc = os.path.join(save_dir, '{}_interventions_N{}.pkl'.format(bif_file.split('.bif')[0], N))
    with open(save_loc,'wb') as f:
        pickle.dump(models, f)
    print('Saved @ {}'.format(save_loc))