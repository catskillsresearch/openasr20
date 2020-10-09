from graphviz import Digraph

diagram = Digraph()
diagram.node('Subsplit <Aijk,Tijk>', shape='note')
diagram.node('ASR NN Trainer', shape='box3d')
diagram.edge('Subsplit <Aijk,Tijk>', 'ASR NN Trainer',color="black:invis:black:invis:black", arrowsize='2')
diagram.node('Model ASR',shape='doubleoctagon')
diagram.edge('ASR NN Trainer', 'Model ASR',color="black", arrowsize='2')
diagram.node('Model output <Gold,Pred>=<Tijk,Pijk>', shape='note')
diagram.edge('ASR NN Trainer','Model output <Gold,Pred>=<Tijk,Pijk>',color="black:invis:black:invis:black", arrowsize='2')
diagram.node('Scoring function',shape='box3d')
diagram.edge('Model output <Gold,Pred>=<Tijk,Pijk>','Scoring function',color="black:invis:black:invis:black", arrowsize='2')
diagram.node('Scores WER<Tijk,Pijk>,CER<Tijk,Pijk>', shape='note')
diagram.edge('Scoring function','Scores WER<Tijk,Pijk>,CER<Tijk,Pijk>',color="black:invis:black:invis:black", arrowsize='2')
