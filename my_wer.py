def my_wer(a,b):
    w=wer(a,b)
    return w*len(a.split(' '))
