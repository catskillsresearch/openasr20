from glob import glob

def load_recent(C, model):
    models=list(sorted(glob('save/nemo_{C.language}/*.ckpt')))
    if len(models):
        last_fn=models[-1]
        model.load_from_checkpoint(last_fn)
        print('loaded', last_fn)
