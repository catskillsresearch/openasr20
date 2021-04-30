import os, glob
from tqdm.auto import tqdm

class Cfg:

    def __init__(self, _stage, _sample_rate, _language, _phase='build', _release='001'):
        self.stage = _stage
        self.sample_rate = _sample_rate
        self.language = _language
        self.phase = _phase
        self.release = _release
        self.data_dir=f'{self.stage}/openasr20_{self.language}'
        self.build_dir=f'{self.data_dir}/{self.phase}'
        self.audio_split_dir=f'{self.build_dir}/audio_split'
        os.system(f'mkdir -p {self.build_dir}')
        os.system(f'mkdir -p {self.audio_split_dir}')
        self.shipping_dir=f'ship/{self.language}/{self.phase}/{self.release}'
        self.model_save_dir=f'save/nemo_{self.language}'

    def transcription_to_stm(self):
        tdir=f'{self.build_dir}/transcription_stm'
        os.system(f'mkdir -p {tdir}')
        tfns=glob.glob(f'{self.build_dir}/transcription/*.txt')
        for txt_file in tfns:
            cmd = f'python OpenASR_convert_reference_transcript.py -f {txt_file} -o {tdir}'
            print("cmd", cmd)
            os.system(cmd)
        
    def split_files(self):
        files = glob.glob(f'{self.audio_split_dir}/*.wav')
        D = []
        for fn in files:
            key=os.path.basename(fn)[0:-4].split('_')
            ctm='_'.join(key[0:7])
            F='_'.join(key[0:6])
            channel=key[6]
            tstart=float(key[-2])
            tend=float(key[-1])
            tbeg=tstart/self.sample_rate
            tdur=(tend-tstart)/self.sample_rate
            D.append({'name': fn, 'key': key, 'channel': channel,
                      'start': int(tstart), 'end': int(tend),
                      't_seconds': tdur, 't_begin': tbeg})
        return D
    
