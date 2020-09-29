import audioread
import librosa
import soundfile as sf

def load_and_resample_if_necessary(_config, audio_file):
    with audioread.audio_open(audio_file) as f:
        sr = f.samplerate
    x_np,sr1=librosa.load(audio_file, sr=_config.sample_rate)
    if sr != _config.sample_rate:
        sf.write(audio_file, x_np, _config.sample_rate)
    return x_np
