# coding: utf-8
# ASR post-processing corrector pred vs gold: TRAIN

import torch
from afterburner_pretrained_model import afterburner_pretrained_model
import matplotlib.pyplot as plt    
from progress_bar import progress_bar
from tqdm.auto import tqdm
import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def afterburner_train(language, phase, release, model_fn, new_model_fn, epochs, batch_size=32):
    C, model, SRC, TRG, device, train_iterator,_ = afterburner_pretrained_model(language, phase, release, model_fn, batch_size)
    LEARNING_RATE = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = torch.nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    model.train();
    print(f'{len(train_iterator)} batches / epoch')
    epoch_loss = 9999999999999999
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    losses = []
    for j in tqdm(range(epochs)):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_iterator)):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            optimizer.zero_grad()
            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        progress_bar(fig, ax, losses)
        torch.save(model.state_dict(), f'{new_model_fn}.{j}')
    torch.save(model.state_dict(), new_model_fn)

if __name__=="__main__":
    language='vietnamese'
    phase='build'
    release='400'
    model_fn='save/new_afterburner/afterburner_302.pt'
    new_model_fn='save/new_afterburner/afterburner_400.pt'
    epochs = 1000
    afterburner_train(language, phase, release, model_fn, new_model_fn, epochs)
