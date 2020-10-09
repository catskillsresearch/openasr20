from graphviz import Digraph

diagram = Digraph()
diagram.node('Model output <Pijk>', shape='note')
diagram.node('Concatenate',shape='box3d')
diagram.node('Model output <Pij>', shape='note')
diagram.edge('Model output <Pijk>','Concatenate',color="black:invis:black:invis:black", arrowsize='2')
diagram.edge('Concatenate','Model output <Pij>',color="black:invis:black:invis:black", arrowsize='2')

