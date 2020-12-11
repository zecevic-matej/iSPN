from spn.io.Text import str_to_spn
from spn.algorithms.Inference import likelihood
import numpy as np

with open('spn_observational.txt','r') as f:
    str_spn = f.readlines()
str_spn = ''.join(str_spn)

model = str_to_spn(str_spn)

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