#!/usr/bin/env python
# coding: utf-8

import torch

def silence_detector(sample_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H,D_in,D_out=sample_rate,sample_rate,sample_rate
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.ReLU()
    ).cuda()
    model_fn='silence_detector.pt'
    model.load_state_dict(torch.load(model_fn))
    return model
