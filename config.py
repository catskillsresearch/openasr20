import os

class Config:

    def __init__(self):
        self.stage='NIST'
        self.sample_rate=8000
        self.batch_size=12
        self.n_epochs=3
        self.save_every = 1

    def update(self):
        self.model_name=f'{self.language}_{self.sample_rate}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4'
        self.model_dir=f'save/{self.model_name}'
        self.best_model=f'{self.model_dir}/best_model.th'
        self.grapheme_dictionary_fn = f'analysis/{self.language}/{self.language}_characters.json'
        self.build_dir=f'{self.stage}/openasr20_{self.language}/build'
        self.split_dir=f"{self.stage}/openasr20_{self.language}/build/audio_split"
        self.nr_dir=f"{self.split_dir}_{self.sample_rate}"
        os.system(f"mkdir -p {self.model_dir}");
        os.system(f"mkdir -p analysis/{self.language}");
        self.stm_dir=f'{self.stage}/openasr20_{self.language}/build/transcription_stm'
        os.system(f'mkdir -p {self.stm_dir}')

C=Config()


