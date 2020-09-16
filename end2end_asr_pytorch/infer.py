import json
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh
from utils.functions import save_model, load_model
from utils.lstm_utils import LM

def evaluate(f, model, test_loader, lm=None):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()

    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            src, tgt, src_percentages, src_lengths, tgt_lengths = data

            if constant.USE_CUDA:
                src = src.cuda()
                tgt = tgt.cuda()

            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, tgt, beam_search=constant.args.beam_search,
                beam_width=constant.args.beam_width, beam_nbest=constant.args.beam_nbest, lm=lm,
                lm_rescoring=constant.args.lm_rescoring, lm_weight=constant.args.lm_weight,
                c_weight=constant.args.c_weight, verbose=constant.args.verbose)

            for x in range(len(batch_strs_hyps)):
                hyp = batch_strs_hyps[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.PAD_CHAR, "")
                print("HYP ", hyp)
                f.write(f'{hyp}\n')
                f.flush()
                
if __name__ == '__main__':

    args = constant.args

    start_iter = 0

    # Load the model
    load_path = constant.args.continue_from
    model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(constant.args.continue_from)
    
    if loaded_args.parallel:
        print("unwrap data parallel")
        model = model.module

    audio_conf = dict(sample_rate=loaded_args.sample_rate,
                      window_size=loaded_args.window_size,
                      window_stride=loaded_args.window_stride,
                      window=loaded_args.window,
                      noise_dir=loaded_args.noise_dir,
                      noise_prob=loaded_args.noise_prob,
                      noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

    test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=constant.args.test_manifest_list, label2id=label2id,
                                   normalize=True, augment=False)
    test_sampler = BucketingSampler(test_data, batch_size=constant.args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

    lm = None
    if constant.args.lm_rescoring:
        lm = LM(constant.args.lm_path)

    fn = constant.args.output
    print("output file: ", fn)
    
    with open(fn, 'w', encoding="utf-8") as f:
         evaluate(f, model, test_loader, lm=lm)
    print('saved', fn)
