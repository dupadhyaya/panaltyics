# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:03:22 2017 by Dhiraj Upadhyaya
Dictionary
"""

# Create an empty dictionary for associating radish names
# with vote counts
counts = {}

for line in open("./data/radishsurvey.txt"):
    line = line.strip()
    name, vote = line.split(" - ")
    if vote not in counts:
        # First vote for this variety
        counts[vote] = 1
    else:
        # Increment the vote count
        counts[vote] = counts[vote] + 1
print(counts)

for name in counts:
    count = counts[name]
    print(name, count, sep=' ', end='\n')
    print(name + ": " + str(count) )  # was giving error to variable assigned earlier

    