#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from Cfg import Cfg
C = Cfg('NIST', 16000, 'pashto', 'dev') 


# In[3]:


from RecordingCorpus import RecordingCorpus
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':   
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)


# In[4]:


from SplitCorpus import SplitCorpus
splits=SplitCorpus.split_on_silence(C, recordings, 33)


# In[5]:


import os
os.system(f'rm {C.audio_split_dir}/*.wav')


# In[6]:


from tqdm.auto import tqdm
import soundfile as sf
for artifact in tqdm(splits.artifacts):
    language, root, start, end = artifact.key
    filename = f'{C.audio_split_dir}/{root}_{start}_{end}.wav'
    audio = artifact.source.value
    sf.write(filename, audio, C.sample_rate)


# In[7]:


vars(C)


# In[ ]:




