# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:23:03 2017 by Dhiraj Upadhyaya
"""

#barplot

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
n_groups = 5

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)
type(means_men)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()
fig

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='b',yerr=std_men, error_kw=error_config, label='Men')

rects2 = plt.bar(index + bar_width, means_women, bar_width,alpha=opacity, color='r', yerr=std_women, error_kw=error_config, label='Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
plt.legend()

plt.tight_layout()
plt.show()

#Egs
df = pd.read_csv('dsstudents2.csv')
df
df.loc[:, lambda df: ['sql', 'stats']]

ax= df[['sql','stats']].plot(kind='bar', title ="V comp",figsize=(15,10),legend=True, fontsize=12)
xlabels = df[['name']]
plt.xticks(index + bar_width / 2, df[['rollno']])
df.index
ax.set_xlabel('Students', fontsize=12)
ax.set_ylabel("Marks", fontsize=12)
plt.show()
df

plt.figure()
df.columns
df[['sql']].plot(kind='bar')
df[['sql']].plot()
df.iloc[2]
plt.plot(df[['sql']])
plt.show()
