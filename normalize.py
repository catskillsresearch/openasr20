def normalize(audio):
    mean=audio.mean()
    audio = audio-mean
    max=audio.max()
    if max:
        audio = audio/max
    return audio
