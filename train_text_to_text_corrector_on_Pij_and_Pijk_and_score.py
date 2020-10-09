from graphviz import Digraph

diagram = Digraph()
diagram.node('Training set <Pijk,Tijk>', shape='note')
diagram.node('Training set <Pij,Tij>', shape='note')
diagram.node('Union', shape='note')
diagram.node('TT NN Trainer', shape='box3d')
diagram.node('Model TT',shape='doubleoctagon')
diagram.node('Model output <Gold,Pred>=<P,Q>', shape='note')
diagram.node('Associate P to T', shape='box3d')
diagram.node('Scoring input <Gold,Pred>=<T,Q>')
diagram.node('Scoring function',shape='box3d')
diagram.node('Scores WER<T,Q>,CER<T,Q>', shape='note')
for x,y in [('Training set <Pijk,Tijk>', 'Union'),
            ('Training set <Pij,Tij>', 'Union'),
            ('Union','TT NN Trainer'),
            ('TT NN Trainer','Model TT'),
            ('TT NN Trainer','Model output <Gold,Pred>=<P,Q>'),
            ('Model output <Gold,Pred>=<P,Q>','Associate P to T'),
            ('Training set <Pijk,Tijk>', 'Associate P to T'),
            ('Training set <Pij,Tij>', 'Associate P to T'),
            ('Associate P to T','Scoring input <Gold,Pred>=<T,Q>'),
            ('Scoring input <Gold,Pred>=<T,Q>','Scoring function'),
            ('Scoring function','Scores WER<T,Q>,CER<T,Q>')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
