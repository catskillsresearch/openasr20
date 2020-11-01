# Afterburner: Correct ASR pred output to match ASR ground truth output

import pandas as pd
from tqdm.auto import tqdm
from afterburner_pretrained_model import afterburner_pretrained_model
from prediction_to_string import prediction_to_string
from calculate_cer import calculate_cer
from calculate_wer import calculate_wer

def afterburner_eval(language, phase, release, model_fn):
    C, model, SRC, TRG, device, train_iterator, batch_size = afterburner_pretrained_model(language, phase, release, model_fn)
    model.eval();
    R=[]
    for i, batch in enumerate(tqdm(train_iterator)):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)   
        prediction=prediction_to_string(TRG, batch_size, output, False)
        gold=prediction_to_string(TRG, batch_size, trg, True)   
        for hyp,au in zip(prediction, gold):
            R.append((au,hyp,calculate_cer(hyp, au),calculate_wer(hyp, au)))

    R=[(au.strip(), hyp.strip(), cer, wer) for au, hyp, cer, wer in R if '<pad>' not in au]

    results=pd.DataFrame(R, columns=['Gold', 'Pred', 'CER', 'WER'])
    results['GOLD_n_words']=results['Gold'].apply(lambda x: len(x.split(' ')))
    results['GOLD_n_chars']=results['Gold'].apply(lambda x: len(x))
    results['CER_pct']=results.CER/results['GOLD_n_chars']
    results['WER_pct']=results.CER/results['GOLD_n_words']
    results=results[results.Gold != '<pad>']

    print('mean WER', results.WER_pct.mean(), 'mean CER', results.CER_pct.mean())

if __name__=="__main__":
    language='vietnamese'
    phase='build'
    release='b30'
    model_fn='save/new_afterburner/afterburner_302.pt'
    afterburner_eval(language, phase, release, model_fn)
