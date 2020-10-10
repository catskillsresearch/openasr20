import os
from tempfile import NamedTemporaryFile
import soundfile as sf
from utils.audio import load_audio

def augment_audio_with_sox_RAM(audio, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as source_file:
        sf.write(source_file, audio, sample_rate)
        path = source_file.name
        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = ["tempo", "{:.3f}".format(
                tempo), "gain", "{:.3f}".format(gain)]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(
                path, sample_rate, augmented_filename, " ".join(sox_augment_params))
            os.system(sox_params)
            y = load_audio(augmented_filename)

    return y
