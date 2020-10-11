from graphviz import Digraph

diagram = Digraph()
diagram.node('Training set <Pijk,Tijk>', shape='note')
diagram.node('Training set <Pij,Tij>', shape='note')
diagram.node('Union', shape='note')
diagram.node('Training set <P,T>', shape='note')
for x,y in [('Training set <Pijk,Tijk>', 'Union'),
            ('Training set <Pij,Tij>', 'Union'),
            ('Union','Training set <P,T>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
