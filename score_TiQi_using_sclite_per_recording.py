from graphviz import Digraph

diagram = Digraph()
diagram.node('Ti.CTM', shape='folder')
diagram.node('Qi.CTM', shape='folder')
diagram.node('sclite', shape='box3d')
diagram.node('CERi,WERi', shape='note')
for x,y in [('Ti.CTM', 'sclite'),
            ('Qi.CTM', 'sclite'),
            ('sclite', 'CERi,WERi')]:
    diagram.edge(x,y,color="black", arrowsize='1')
