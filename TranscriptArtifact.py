import os
import pandas as pd
from txt_to_stm import txt_to_stm
from Artifact import Artifact

class TranscriptArtifact (Artifact):

    def __init__(self, _config, _filename):
        file = "_".join(os.path.basename(_filename).split("_")[:-1])
        channel = os.path.basename(_filename).split("_")[-1].split(".")[-2]
        transcript_df = pd.read_csv(_filename, sep = "\n", header = None, names = ["content"])
        no_Q = (_config.language == 'cantonese')
        _value = txt_to_stm(transcript_df, file, channel, no_Q)
        super().__init__(_config, _value)
        self.filename = _filename
        self.transcript = transcript_df

    def display(self):
        return pd.DataFrame(self.value, columns=['filename', 'channel', 'both', 'start', 'end', 'text'])


