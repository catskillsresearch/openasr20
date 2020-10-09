from torch.utils.data import Dataset
import numpy as np
import os
import subprocess
import torchaudio
from tempfile import NamedTemporaryFile
from load_randomly_augmented_audio_RAM import load_randomly_augmented_audio_RAM

class SpectrogramDatasetRAM(Dataset):

    def __init__(self, audio_conf, corpus, label2id, normalize=False, augment=False):
        """
        Dataset that loads tensors via a corpus of (audio, text) pairs.
        Parses audio into spectrogram with optional normalization and various augmentations
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param corpus: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.normalize = normalize
        self.augment = augment
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')
        self.label2id = label2id
        self.corpus = [(self.parse_audio(audio)[:,:constant.args.src_max_len], self.parse_transcript(text)) for audio, text in corpus]
        self.max_size = len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]

    def parse_audio(self, audio):
        if self.augment:
            y = load_randomly_augmented_audio_RAM(audio_path, self.sample_rate)
        else:
            y = audio

        if self.noiseInjector:
            logging.info("inject noise")
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        # Short-time Fourier transform (STFT)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, _transcript):
        transcript = constant.SOS_CHAR + _transcript.replace('\n', '').lower() + constant.EOS_CHAR
        transcript = list(filter(None, [self.label2id.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.max_size
