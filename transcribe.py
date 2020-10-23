import soundfile as sf

def transcribe(C, model, audio):
    fn='tmp.wav'
    sf.write(fn, audio, C.sample_rate)
    translations=model.transcribe(paths2audio_files=[fn], batch_size=1)
    translation=translations[0]
    translation=translation.split(' ')
    translation=' '.join([x.strip() for x in translation if len(x)])
    return translation.replace("\u200c",'')  # Just Pashto but required
