from IPython.display import Audio, display

def play(sound):
    display(Audio(sound, rate=8000))
