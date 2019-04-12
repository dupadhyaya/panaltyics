# -*- coding: utf-8 -*-
"""
Sets
"""
S1 = set([1, 2, 3, 4])
S2 = set([3, 4, 5, 6])
S3 = S1.intersection(S2)
S3
S1.intersection(S2)
S1.difference(S2)
S2.difference(S1)

S3.issubset(S1)

S3.union(S1)
S3.isdisjoint(S1)

S3.issuperset(S1)
S1.issuperset(S3)


S1.difference(S3)
S1.symmetric_difference(S2)
S1.copy()
