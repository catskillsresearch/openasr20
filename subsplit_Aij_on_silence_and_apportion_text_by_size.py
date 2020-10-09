from graphviz import Digraph

diagram = Digraph()
diagram.node('Split <Aij,Tij>', shape='tab')
diagram.node('Subsplit on silence and apportion text to trimmed chunks by word size', shape='box3d')
diagram.node('Subsplit <Aijk,Tijk>', shape='note')
diagram.edge('Split <Aij,Tij>', 'Subsplit on silence and apportion text to trimmed chunks by word size',color="black:invis:black:invis:black", arrowsize='2')
diagram.edge('Subsplit on silence and apportion text to trimmed chunks by word size', 'Subsplit <Aijk,Tijk>', color="black:invis:black:invis:black", arrowsize='2')

