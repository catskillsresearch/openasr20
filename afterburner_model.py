import numpy as np
import os, random, torch
from count_parameters import count_parameters
from initialize_weights import initialize_weights
from torchtext.data import Field
from seq_to_seq import *

def afterburner_model(graphemes, MAX_LENGTH, model_fn = None):
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    tokenize=lambda x: [y for y in x]

    SRC = Field(tokenize = tokenize, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    TRG = Field(tokenize = tokenize, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    ## Model

    MIN_FREQ=1

    SRC.build_vocab(graphemes, min_freq = MIN_FREQ)

    TRG.build_vocab(graphemes, min_freq = MIN_FREQ)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    INPUT_DIM, OUTPUT_DIM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, 
                  HID_DIM, 
                  ENC_LAYERS, 
                  ENC_HEADS, 
                  ENC_PF_DIM, 
                  ENC_DROPOUT, 
                  device,
                  MAX_LENGTH)

    dec = Decoder(OUTPUT_DIM, 
                  HID_DIM, 
                  DEC_LAYERS, 
                  DEC_HEADS, 
                  DEC_PF_DIM, 
                  DEC_DROPOUT, 
                  device,
                  MAX_LENGTH)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.apply(initialize_weights);
    if model_fn is not None and os.path.exists(model_fn):
        model.load_state_dict(torch.load(model_fn))
        print('NOTE: Reloaded trained model', model_fn)
    else:
        print("WARNING: could not find", model_fn)
        
    return model, SRC, TRG, device
