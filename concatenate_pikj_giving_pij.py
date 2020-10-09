from graphviz import Digraph

diagram = Digraph()
diagram.node('Model Interence <Pijk,(sijk,eijk)>', shape='note')
diagram.node('Concatenate', shape='box3d')
diagram.node('Model Inference <Pij, (sij0,eij-1)>', shape='note')
for x,y in [('Model Interence <Pijk,(sijk,eijk)>', 'Concatenate'),
        ('Concatenate', 'Model Inference <Pij, (sij0,eij-1)>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')

