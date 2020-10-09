from graphviz import Digraph

def diagram(C):
    dot = Digraph()
    dot.node(f'Language {C.language} BUILD corpus', shape='cylinder')
    dot.node('Load .WAV and .TSV files', shape='box3d')
    dot.edge(f'Language {C.language} BUILD corpus', 'Load .WAV and .TSV files', color="black:invis:black:invis:black", arrowsize='2')
    dot.node('Recording <Ai,Ti>', shape='folder')
    dot.edge('Load .WAV and .TSV files', 'Recording <Ai,Ti>', color="black:invis:black:invis:black", arrowsize='2')
    dot.node('Apply transcript splits', shape='box3d')
    dot.edge('Recording <Ai,Ti>', 'Apply transcript splits', color="black:invis:black:invis:black", arrowsize='2')
    dot.node('Split <Aij,Tij>', shape='tab')
    dot.edge('Apply transcript splits', 'Split <Aij,Tij>', color="black:invis:black:invis:black", arrowsize='2')
    return dot
