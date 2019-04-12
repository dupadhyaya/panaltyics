# -*- coding: utf-8 -*-
"""
Wed May  9 17:58:19 2018: Dhiraj
"""
#http://graphviz.readthedocs.io/en/stable/manual.html
#working now : had to restart Spyder and PC after installing and configuring path for graphviz

from graphviz import Digraph
dot = Digraph(comment='The Round Table')

dot  
print(dot)
#
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')
print(dot.source) 

dot.render('test-output/round-table.gv', view=True)

#
from graphviz import Graph
g = Graph(format='png')
dot.format = 'svg'
dot.render()  


#Piped Output
h = Graph('hello', format='svg')
h.edge('Hello', 'World')
print(h.pipe().decode('utf-8')) 
