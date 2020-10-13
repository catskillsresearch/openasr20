import sys, json
sys.path.append('/home/catskills/Desktop/openasr20/end2end_asr_pytorch')
from utils import constant
from utils.functions import save_model, load_model, init_transformer_model, init_optimizer
from TrainerVanilla import TrainerVanilla
from utils.data_loader import AudioDataLoader
from SpectrogramDatasetRAM import SpectrogramDatasetRAM
import logging
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
from initialize_weights import initialize_weights

class FucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(FucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def shuffle(self, epoch):
       pass
   
    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)


class ASR_NN:

    def __init__(self, C):
        self.config=C
        self.trainer = TrainerVanilla()
        self.config.extension='_gradscaler'
        self.config.batch_size=12
        self.config.save_every = 1
        self.config.start_from = 246
        self.config.analysis_dir = f'analysis/{self.config.language}'
        self.config.grapheme_dictionary_fn = f'{self.config.analysis_dir}/{self.config.language}_characters.json'
        self.config.model_name=f'{self.config.language}_{self.config.sample_rate}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4{self.config.extension}'
        self.config.model_dir=f'save/{self.config.model_name}'
        self.config.best_model=f'{self.config.model_dir}/best_model.th'
        constant.USE_CUDA=True
        constant.args.save_every = 1
        constant.args.continue_from=None
        constant.args.cuda = True
        constant.args.labels_path = self.config.grapheme_dictionary_fn
        constant.args.lr = 1e-4
        constant.args.name = self.config.model_name
        constant.args.save_folder = f'save'
        constant.args.epochs = 1000
        constant.args.save_every = 1
        constant.args.feat_extractor = f'vgg_cnn'
        constant.args.dropout = 0.1
        constant.args.num_layers = 4
        constant.args.num_heads = 8
        constant.args.dim_model = 512
        constant.args.dim_key = 64
        constant.args.dim_value = 64
        constant.args.dim_input = 161
        constant.args.dim_inner = 2048
        constant.args.dim_emb = 512
        constant.args.shuffle=True
        constant.args.min_lr = 1e-6
        constant.args.k_lr = 1
        constant.args.sample_rate=self.config.sample_rate
        constant.args.continue_from=self.config.best_model
        constant.args.augment=False
        self.audio_conf = dict(sample_rate=constant.args.sample_rate,
                               window_size=constant.args.window_size,
                               window_stride=constant.args.window_stride,
                               window=constant.args.window,
                               noise_dir=constant.args.noise_dir,
                               noise_prob=constant.args.noise_prob,
                               noise_levels=(constant.args.noise_min, constant.args.noise_max))

    def new_model(self):
        with open(constant.args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))
        # add PAD_CHAR, SOS_CHAR, EOS_CHAR
        labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
        self.label2id, self.id2label = {}, {}
        count = 0
        for i in range(len(labels)):
            if labels[i] not in self.label2id:
                self.label2id[labels[i]] = count
                self.id2label[count] = labels[i]
                count += 1
            else:
                print("multiple label: ", labels[i])
        self.model = init_transformer_model(constant.args, self.label2id, self.id2label)
        self.opt = init_optimizer(constant.args, self.model, "noam")
        self.model.cuda(0)

    def load_model(self):
        self.config.extension='_gradscaler'
        self.config.model_name=f'{self.config.language}_{self.config.sample_rate}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4{self.config.extension}'
        self.config.model_dir=f'save/{self.config.model_name}'
        self.config.best_model=f'{self.config.model_dir}/best_model.th'
        constant.args.continue_from=self.config.best_model
        logging.info(f'load {constant.args.continue_from}')
        self.model, self.opt, self.epoch, self.metrics, self.loaded_args, self.label2id, self.id2label = load_model(constant.args.continue_from)
        self.model = self.model.cuda(0)
        logging.info(self.model)

    def load_training_set(self, subsplits, batch_size = 1):
        constant.args.batch_size=batch_size
        corpus = [(artifact.source.value, artifact.target.value) for artifact in subsplits.artifacts]
        train_data = SpectrogramDatasetRAM (self.audio_conf, corpus, self.label2id, normalize=False, augment=False)
        self.train_sampler = FucketingSampler(train_data, batch_size=constant.args.batch_size)
        self.train_loader = AudioDataLoader(train_data, num_workers=constant.args.num_workers, batch_sampler=self.train_sampler)
        
    def train(self):
        (self.results, self.metrics) = self.trainer.train(self.model,
                                  self.train_loader,
                                  self.train_sampler,
                                  self.opt,
                                  constant.args.loss,
                                  0, 1,
                                  self.label2id, self.id2label, just_once=True)
        return self.results

    def save(self, filename):
        save_model(self.model, 0, self.opt, self.metrics, self.label2id, self.id2label, save_path = filename)

    def infer(self):
        return self.trainer.infer(self.model,
                                  self.train_loader,
                                  self.train_sampler,
                                  self.label2id, self.id2label)


    def score(self, output):
        import pandas as pd
        df=pd.DataFrame([x for x in output if len(x[1]) > 0], columns=['order', 'hyp', 'gold', 'cer', 'wer'])
        df['gold_chars']=df.gold.apply(len)
        df['gold_words']=df.gold.apply(lambda x: len(x.split(' ')))
        df['cer_pct']=df.cer/(df.gold_chars+0.0000001)
        df['wer_pct']=df.wer/(df.gold_words+0.000001)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
        df.cer_pct.plot(kind='hist',bins=100, edgecolor='black', linewidth=1.2, ax=axes[0])
        axes[0].set_title(r'CER %')
        df.wer_pct.plot(kind='hist',bins=100, edgecolor='black', linewidth=1.2, ax=axes[1])
        axes[1].set_title(r'WER %')
        return df
