import random, torch, warnings, sys, os, json, logging
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from prediction_to_string import prediction_to_string
from progress_bar import progress_bar
from tqdm.notebook import tqdm
sys.path.append('/home/catskills/Desktop/openasr20/end2end_asr_pytorch')
os.environ['IN_JUPYTER']='True'
from utils.metrics import calculate_cer, calculate_wer
from initialize_weights import initialize_weights
from count_parameters import count_parameters
from seq_to_seq import *
from torchtext.data import Field, Iterator, TabularDataset
from tempfile import NamedTemporaryFile

warnings.filterwarnings("ignore")

class TTC_NN:

    def __init__(self, _config):
        self.config=_config
        self.set_seed()
        tokenize=lambda x: [y for y in x]
        self.SRC = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
        self.TRG = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
        self.HID_DIM = 256
        self.ENC_LAYERS = 3
        self.DEC_LAYERS = 3
        self.ENC_HEADS = 8
        self.DEC_HEADS = 8
        self.ENC_PF_DIM = 512
        self.DEC_PF_DIM = 512
        self.ENC_DROPOUT = 0.1
        self.DEC_DROPOUT = 0.1
        self.MAX_LENGTH=512
        with open(f'analysis/{self.config.language}/{self.config.language}_characters.json', 'r', encoding='utf-8') as f:
            self.graphemes=list(sorted(json.load(f)))
        self.SRC.build_vocab(self.graphemes, min_freq = 1)
        self.TRG.build_vocab(self.graphemes, min_freq = 1)
        self.INPUT_DIM = len(self.SRC.vocab)
        self.OUTPUT_DIM = len(self.TRG.vocab)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc = Encoder(self.INPUT_DIM, self.HID_DIM, self.ENC_LAYERS, self.ENC_HEADS, 
                           self.ENC_PF_DIM, self.ENC_DROPOUT, self.device,self.MAX_LENGTH)
        self.dec = Decoder(self.OUTPUT_DIM, self.HID_DIM, self.DEC_LAYERS, self.DEC_HEADS, 
                           self.DEC_PF_DIM, self.DEC_DROPOUT, self.device,self.MAX_LENGTH)
        self.SRC_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        self.TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]
        self.model = Seq2Seq(self.enc, self.dec, self.SRC_PAD_IDX, self.TRG_PAD_IDX, self.device).to(self.device)
        self.LEARNING_RATE = 0.0005
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.TRG_PAD_IDX)
        self.j=0

    def set_seed(self):
        SEED = 7
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    def load_model(self, _model_fn = None):
        self.model_fn=_model_fn
        print(f'The model has {count_parameters(self.model):,} trainable parameters')
        if self.model_fn is not None:
            self.model.load_state_dict(torch.load(self.model_fn))
        else:
            self.model.apply(initialize_weights);

    def save_model(self):
        get_ipython().system('mkdir -p save/afterburner')
        self.model_fn=f'save/afterburner/afterburner_002_{os.getpid()}_{self.j}.pt'
        torch.save(self.model.state_dict(), self.model_fn)
        print("saved", self.model_fn)
        
    def load_training_set(self, _PT, _batch_size = 638):
        self.batch_size = _batch_size
        logging.info(f'Batch size {self.batch_size}')
        logging.info(f'#Training examples {len(_PT)}')
        training='\n'.join([f'{pred.strip()[0:self.MAX_LENGTH-1]}\t{gold.strip()[0:self.MAX_LENGTH-1]}' for pred,gold,art in _PT])
        self.graphemes=list(sorted(set([x for x in training if x not in ['\n', '\t']])))
        logging.info(f"#graphemes {len(self.graphemes)}")
        fields = [('src',self.SRC), ('trg',self.TRG)]
        with NamedTemporaryFile(suffix=".wav") as temp_file:
            with open(temp_file.name, 'w', encoding='utf-8') as f:
                f.write(training)
            self.train_data=TabularDataset(path=temp_file.name,format='tsv',fields=fields)
        self.train_iterator = Iterator(self.train_data, batch_size=self.batch_size)

    def train(self, _loss_threshold = 0.1):
        self.model.train();
        print(f'{len(self.train_iterator)} batches / epoch')
        epoch_loss = 9999999999999999
        fig,ax = plt.subplots(1,1)
        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')
        plt.show()
        losses = []
        while epoch_loss > _loss_threshold:
            epoch_loss = 0
            for i, batch in enumerate(self.train_iterator):
                src = batch.src.to(self.device)
                trg = batch.trg.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(src, trg[:,:-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                epoch_loss += loss.item()
            self.j += 1
            losses.append(epoch_loss)
            progress_bar(fig, ax, losses)
            self.save_model()

    def infer(self):
        self.model.eval();
        R=[]
        train_iterator = Iterator(self.train_data, batch_size=self.batch_size)
        for i, batch in enumerate(tqdm(train_iterator)):
            src = batch.src.to(self.device)
            trg = batch.trg.to(self.device)
            with torch.no_grad():
                output, _ = self.model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)   
            prediction=prediction_to_string(self.TRG, self.batch_size, output, False)
            gold=prediction_to_string(self.TRG, self.batch_size, trg, True)
            for hyp,au in zip(prediction, gold):
                if '<pad>' in au:
                    continue
                R.append((au,hyp,calculate_cer(hyp, au),calculate_wer(hyp, au)))
        self.predictions = R
        return R

    def score(self, R):
        results=pd.DataFrame(R, columns=['Gold', 'Pred', 'CER', 'WER'])
        results['GOLD_n_words']=results['Gold'].apply(lambda x: len(x.split(' ')))
        results['GOLD_n_chars']=results['Gold'].apply(lambda x: len(x))
        results['CER_pct']=results.CER/results['GOLD_n_chars']
        results['WER_pct']=results.CER/results['GOLD_n_words']
        results=results[results.Gold != '<pad>']
        results.WER_pct.hist(bins=1000)
#        plt.xlim(0,1)
        print("CER mean", results.CER_pct.mean())
        print("WER mean", results.WER_pct.mean())
        return results
