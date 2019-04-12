# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:53:47 2018
Dhiraj
@author: du
"""

import hello
hello.sayhello("Dhiraj", "Upadhyaya")

# Script to check reload. Do changes in hello module first and run this
import imp
imp.reload(hello)

hello.sayhello("Dhiraj", "Upadhyaya", "Dean DS")


