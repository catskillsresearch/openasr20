from graphviz import Digraph

diagram = Digraph()
diagram.node('Model Inference <Pij, (sij0,eij-1)>', shape='note')
diagram.node('Model TT',shape='doubleoctagon')
diagram.node('TT NN Inference', shape='box3d')
diagram.node('Model Inference <Qij, (sij0,eij-1)>', shape='note')
for x,y in [('Model Inference <Pij, (sij0,eij-1)>', 'TT NN Inference'),
            ('Model TT', 'TT NN Inference'),
            ('TT NN Inference', 'Model Inference <Qij, (sij0,eij-1)>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
