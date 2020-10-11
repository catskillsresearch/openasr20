from graphviz import Digraph

diagram = Digraph()
diagram.node('Training set <P,T>', shape='note')
diagram.node('TTC NN Trainer', shape='box3d')
diagram.node('Model TTC',shape='doubleoctagon')
diagram.node('Model output <Gold,Pred>=<T,Q>', shape='note')
diagram.node('Scoring function',shape='box3d')
diagram.node('Scores WER<T,Q>,CER<T,Q>', shape='note')
for x,y in [('Training set <P,T>','TTC NN Trainer'),
            ('TTC NN Trainer','Model TT'),
            ('TTC NN Trainer','Model output <Gold,Pred>=<T,Q>'),
            ('Model output <Gold,Pred>=<T,Q>','Scoring function'),
            ('Scoring function','Scores WER<T,Q>,CER<T,Q>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
