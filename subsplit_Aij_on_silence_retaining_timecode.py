from graphviz import Digraph
diagram = Digraph()
diagram.node('Audio split <Aij, (Sij,Eij)>', shape='tab')
diagram.node('Recursive split on silence to 2 words', shape='box3d')
diagram.node('Audio subsplits <aijk,(sijk,eijk)>', shape='note')
diagram.edge('Audio split <Aij, (Sij,Eij)>', 'Recursive split on silence to 2 words', color="black:invis:black:invis:black", arrowsize='2')
diagram.edge('Recursive split on silence to 2 words','Audio subsplits <aijk,(sijk,eijk)>', color="black:invis:black:invis:black", arrowsize='2')
