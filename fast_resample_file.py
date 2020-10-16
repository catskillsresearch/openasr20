import librosa
import audioread
import soundfile as sf

def fast_resample_file(audio_file):
    sample_rate = 16000
    with audioread.audio_open(audio_file) as f:
        sr = f.samplerate
    if sr != sample_rate:
        print('RESIZING', sr, audio_file)
        x_np,sr=librosa.load(audio_file, sr=sample_rate)        
        sf.write(audio_file, x_np, sample_rate)
