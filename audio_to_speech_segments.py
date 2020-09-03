#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import librosa, torch, os
import numpy as np
import soundfile as sf
from running_mean import running_mean
from contiguous_regions import contiguous_regions
from to_segments import to_segments

def audio_to_speech_segments(sample_rate, device, model, audio_file):
    audio_dir=os.path.dirname(audio_file)
    audio_split_dir=audio_dir.replace('/audio', '/audio_split')
    Path(audio_split_dir).mkdir(parents=True, exist_ok=True)
    x_np,sr=librosa.load(audio_file, sr=sample_rate)
    recording_length=x_np.shape[0]
    x_samples_np=to_segments(x_np, sample_rate)
    with torch.no_grad():
        tensor_x = torch.Tensor(x_samples_np).to(device)
        y_pred_cuda=model(tensor_x)
    y_pred = np.reshape(y_pred_cuda.cpu().numpy(), -1)
    y_pred = running_mean(y_pred, 500)[0:x_np.shape[0]]
    y_pred = (y_pred < 0.5).astype(float)

    MINIMUM_SEGMENT_SIZE=4000
    regions=[(x,y) for x,y in contiguous_regions(y_pred) if y-x > MINIMUM_SEGMENT_SIZE]
    for start, end in regions:
        clip_fn = f"{audio_file[:-4].replace('/audio/', '/audio_split/')}_{sample_rate}_{start:06d}_{end:06d}.wav"
        sf.write(clip_fn, x_np[start:end], sample_rate)
