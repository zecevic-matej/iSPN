"""
This code snippet was run in conjunction with example_script.py
"""


hits = np.sum(~(mpe - y_batch).any(1))
print("Accuracy {}/{} ({})".format(hits, len(mpe), hits / len(mpe)))

# regular mnist
# 8 / 64 accuracy
# 174.98 abs diff -> 2.73 abs diff on average per prediction

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

avg_ll = [-0.7,-0.61,-0.54,-0.61] # mf, mdn, cspn, cspn (10 epochs)
acc = [74.1, 76.4, 78.4, 77.]

fig, axs = plt.subplots(1,2, sharex=True, figsize=(14,6))
axs[0].bar(range(4), -1*np.array(avg_ll), color=['black', 'black', 'royalblue', 'deepskyblue'])
axs[0].set_title('Average Test Negative Conditional Log-Likelihood')
axs[0].set_ylabel('Score AT NCLL')
axs[0].set_xticks(range(4))
axs[0].set_xticklabels(['MF\n(paper)','MDN\n(paper)','CSPN\n(paper)','CSPN\n(10 epochs reproduction)'])
axs[1].bar(range(4), acc, color=['black', 'black', 'royalblue', 'deepskyblue'])
axs[1].set_title('Accuracy (All 16 labels correct(')
axs[1].set_ylabel('Score Acc')
axs[1].set_ylim(0,100)
plt.suptitle('Reproduction on Modified MNIST Dataset ({0,1}^16)')
plt.show()