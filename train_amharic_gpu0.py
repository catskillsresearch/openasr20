from config import C
from train_language import train_language
C.language='amharic'
C.batch_size=12
C.save_every = 5
C.update()
train_language(C)
