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