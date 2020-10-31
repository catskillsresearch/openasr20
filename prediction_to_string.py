from to_TRG import to_TRG
from eos_trim import eos_trim

def prediction_to_string(TRG, batch_size, output, gold = False):
    pred=output.cpu().detach().numpy()
    sample_length=pred.shape[0]//batch_size
    p2=[pred[i*sample_length:(i+1)*sample_length] for i in range(batch_size)]
    p3=[to_TRG(TRG, x if gold else x.argmax(axis=1)) for x in p2]
    p4=[eos_trim(x) for x in p3]
    return p4

