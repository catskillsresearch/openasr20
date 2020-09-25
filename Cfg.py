class Cfg:

    def __init__(self, _stage, _sample_rate, _language):
        self.stage = _stage
        self.sample_rate = _sample_rate
        self.language = _language
        self.data_dir=f'{self.stage}/openasr20_{self.language}'
        self.build_dir=f'{self.data_dir}/build'
