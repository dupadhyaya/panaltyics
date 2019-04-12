# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:15:11 2018 by Dhiraj Upadhyaya for DS 
http://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php
"""

#gini

import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    return (p)*(1-p) + (1 - p)*(1 - (1-p))


def entropy(p):
    return -p * np.log2(p) - (1-p) * np.log2((1-p))

def classification_error(p):
    return 1- np.max([p, 1-p])

x = np.arange(0.0, 1, 0.01)
x

ent = [ entropy(p) if p!=0 else None for p in x]
scaled_ent = [ e * 0.5 if e else None for e in ent]
c_err = [classification_error(i) for i in x]
c_err

fig = plt.figure()
ax = plt.subplot(111)
for j, lab, ls, c, in zip(
      [ent, scaled_ent, gini(x), c_err],
      ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
      ['-', '-', '--', '-.'],
      ['lightgray', 'red', 'green', 'blue']):
   line = ax.plot(x, j, label=lab, linestyle=ls, lw=1, color=c)

ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85),
         ncol=1, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

plt.ylim([0, 1.1])
plt.xlabel('p(j=1)')
plt.ylabel('Impurity Index')
plt.show()
    