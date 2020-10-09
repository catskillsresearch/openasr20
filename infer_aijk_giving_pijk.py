from graphviz import Digraph

diagram = Digraph()
diagram.node('Audio subsplits <aijk,(sijk,eijk)>', shape='note')
diagram.node('Model ASR',shape='doubleoctagon')
diagram.node('ASR NN Inference', shape='box3d')
diagram.node('Model Interence <Pijk,(sijk,eijk)>', shape='note')
for x,y in [('Audio subsplits <aijk,(sijk,eijk)>', 'ASR NN Inference'),
        ('Model ASR', 'ASR NN Inference'),
        ('ASR NN Inference', 'Model Interence <Pijk,(sijk,eijk)>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
