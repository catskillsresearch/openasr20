from config import C
from train_language import train_language
C.language='amharic'
C.extension='_gradscaler'
C.batch_size=12
C.save_every = 5
C.start_from = 246
C.update()
train_language(C)
