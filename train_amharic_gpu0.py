from config import C
from train_language import train_language
C.language='amharic'
C.batch_size=12
train_language(C)
