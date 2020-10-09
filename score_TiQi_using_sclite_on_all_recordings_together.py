from graphviz import Digraph

diagram = Digraph()
diagram.node('<T1.CTM>', shape='folder')
diagram.node('<Q1.CTM>', shape='folder')
diagram.node('sclite', shape='box3d')
diagram.node('CER,WER', shape='note')
diagram.edge('sclite','CER,WER',color="black", arrowsize='1')
for x,y in [('<T1.CTM>', 'sclite'),
            ('<Q1.CTM>', 'sclite')]:
    diagram.edge(x,y,color="black:invis:black:invis:black", arrowsize='2')
