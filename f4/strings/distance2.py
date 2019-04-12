# -*- coding: utf-8 -*-
"""
Tue Apr 24 13:53:58 2018: Dhiraj
"""
#Distance2
#https://stackoverflow.com/questions/6690739/fuzzy-string-comparison-in-python-confused-with-which-library-to-use

import Levenshtein
Levenshtein.ratio('hello world', 'hello')


import difflib
difflib.SequenceMatcher(None, 'hello world', 'hello').ratio()
