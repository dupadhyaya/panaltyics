# -*- coding: utf-8 -*-
"""
Mon Apr 23 12:20:35 2018: Dhiraj
"""
https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

import numpy as np
import sklearn.cluster
import distance


words ="Deana,Diane,Dionne,Gerald,Irina,Lisette,Minna,Nicki,Ricki"
words2 = "ship banana apple minivan peach bus truck guava train pineapple car"
words3 = "tiger eagle lion hawk humming_bird mackarel hilsa cat parrot owl crow dog"
words = words.split(",") #Replace this line
words = words2.split(" ")
words = words3.split(" ")

list(words)
words
words = np.asarray(words) #So that indexing with a list will work
words
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))
    

t1 = ("de", "ci", "si", "ve")
t2 = ("de", "ri", "si", "ve")
distance.levenshtein(t1, t2)

sent1 = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
sent2 = ['the', 'lazy', 'fox', 'jumps', 'over', 'the', 'crazy', 'dog']
distance.levenshtein(sent1, sent2)

distance.hamming("fat", "cat", normalized=True)
#0.3333333333333333
distance.nlevenshtein("abc", "acd", method=1)  # shortest alignment
#0.6666666666666666
distance.nlevenshtein("abc", "acd", method=2)  # longest alignment
#0.5


distance.sorensen("decide", "resize")
#0.5555555555555556
distance.jaccard("decide", "resize")
#0.7142857142857143
