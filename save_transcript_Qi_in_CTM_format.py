from graphviz import Digraph

diagram = Digraph()
diagram.node('Transcript Qi', shape='folder')
diagram.node('Format as .CTM',shape='box3d'),
diagram.node('Q1.CTM', shape='folder')
for x,y in [('Transcript Qi', 'Format as .CTM'),
            ('Format as .CTM', 'Q1.CTM')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
