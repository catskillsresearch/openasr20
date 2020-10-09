from graphviz import Digraph

diagram = Digraph()
diagram.node('Model Inference <Qij, (sij0,eij-1)>', shape='note')
diagram.node('Concatenate',shape='box3d'),
diagram.node('Transcript Qi', shape='folder')
for x,y in [('Model Inference <Qij, (sij0,eij-1)>', 'Concatenate'),
            ('Concatenate', 'Transcript Qi')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')

