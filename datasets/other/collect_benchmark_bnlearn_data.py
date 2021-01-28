import bnlearn as bn
import numpy as np

# load model
bif_file = 'asia.bif'
model_obs = bn.import_DAG(bif_file)

# visualize graph
visualize = False
if visualize:
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
N = 10000
df_obs = bn.sampling(model_obs, n=N)

# example intervention on tub
def intervene(model, var):
    cpts = model['model'].get_cpds()
    modeled_vars = [c.variable for c in cpts]
    assert(len(modeled_vars) == len(cpts))
    # find cpt of node to be intervened one
    sel_cpt = cpts[modeled_vars.index(var)]
    conditions = sel_cpt.variables.copy()
    conditions.remove(var)
    # remove parents
    for v in conditions:
        model['model'].remove_node(v)
    # overwrite cpt with intervention cpt
    sel_cpt.values = np.array([0., 1.])
    print('\nIntervened on {}'.format(var))
    print_cpds(model)
model_int = model_obs.copy()
intervene(model_int, 'tub')
df_int = bn.sampling(model_int, n=N)
assert(sum(df_obs['tub']) < sum(df_int['tub']))