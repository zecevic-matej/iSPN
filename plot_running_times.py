from glob import glob
import numpy as np

l = glob('runtimes/regular/*/runtimes.txt')

d = {
    'Asia': {
        'MDN': -1,
        'MADE': -1,
        'iSPN': -1,
    },
    'Earthquake': {
        'MDN': -1,
        'MADE': -1,
        'iSPN': -1,
    },
    'Cancer': {
        'MDN': -1,
        'MADE': -1,
        'iSPN': -1,
    }
}
for p in l:
    for k in d:
        if k.lower() in p:
            with open(p, "r") as f:
                g = [float(x.strip()) for x in f.readlines()]
            d[k][p.split("timed_")[1].split("_")[0]] = np.round(np.mean(g), decimals=2)
d.update({
    'CH': {
        'MDN': 348.67,
        'MADE': 529.58,
        'iSPN': 227.34,
    }}
)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

f, axs = plt.subplots(1, 1, figsize=(8, 5))#4, 1, figsize=(8, 5))
palette = sns.color_palette(['#f4c09c','#b582de','#a7da86'])#[np.array([244, 192, 156, 1]), np.array([181, 130, 222, 1]), np.array([167, 218, 134, 1])])
resorted = [3, 0, 1, 2]
#for ind, ax in enumerate(axs):
e = list(d.values())[resorted[0]]
y = np.array(list(e.keys()))
x = np.array(list(e.values()))
y = np.flip(y)
x = np.flip(x)
sns.barplot(y=y, x=x, ax=axs, palette=palette)

# Add a legend and informative axis label
axs.set(ylabel="", xlabel="Mean Training Time in Seconds")
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()