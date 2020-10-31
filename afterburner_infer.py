# Afterburner: Correct ASR pred output to match ASR ground truth output

import pandas as pd
from tqdm.auto import tqdm
from afterburner_model import afterburner_model
from torchtext.data import TabularDataset, Iterator
from calculate_cer import calculate_cer
from calculate_wer import calculate_wer
from prediction_to_string import prediction_to_string

def get_model():
    model_fn='save/new_afterburner/afterburner_300.pt'
    MAX_LENGTH=496
    graphemes=[' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
               'r','s','t','u','v','w','x','y','à','á','â','ã','è','é','ê','ì','í','ò',
               'ó','ô','õ','ù','ú','ý','ă','đ','ĩ','ũ','ơ','ư','ạ','ả','ấ','ầ','ẩ','ẫ',
               'ậ','ắ','ằ','ẳ','ẵ','ặ','ẹ','ẻ','ẽ','ế','ề','ể','ễ','ệ',
               'ỉ','ị','ọ','ỏ','ố','ồ','ổ','ỗ','ộ','ớ','ờ','ở','ỡ','ợ',
               'ụ','ủ','ứ','ừ','ử','ữ','ự','ỳ','ỷ','ỹ']
    model, SRC, TRG, device = afterburner_model(graphemes, MAX_LENGTH, model_fn)
    return model, SRC, TRG, device
    
def afterburner_infer(preds):
    # save input to file
    training='\n'.join([f'{x.strip()}\tx' for x in preds])a
    error_correction_training_fn='infer.tsv'
    with open(error_correction_training_fn, 'w', encoding='utf-8') as f:
        f.write(training)
    # get model
    model, SRC, TRG, device = get_model()
    # make data loader
    train_data = TabularDataset(path=error_correction_training_fn, format='tsv', fields=[('trg', TRG), ('src', SRC)])
    batch_size=32
    train_iterator = Iterator(train_data, batch_size=batch_size)
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
