from config import C
from train_language import train_language
C.language='pashto'
C.batch_size=6
C.save_every = 25
C.update()
train_language(C)

