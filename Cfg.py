import os

class Cfg:

    def __init__(self, _stage, _sample_rate, _language, _phase='build', _release='001'):
        self.stage = _stage
        self.sample_rate = _sample_rate
        self.language = _language
        self.phase = _phase
        self.release = _release
        self.data_dir=f'{self.stage}/openasr20_{self.language}'
        self.build_dir=f'{self.data_dir}/{self.phase}'
        self.audio_split_dir=f'{self.build_dir}/audio_split'
        os.system(f'mkdir -p {self.build_dir}')
        os.system(f'mkdir -p {self.audio_split_dir}')
        self.shipping_dir=f'ship/{self.language}/{self.release}'
