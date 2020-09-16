import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from TrainerVanilla import TrainerVanilla
from utils.optimizer import NoamOpt
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_optimizer
import logging

import sys
import os

if __name__ == '__main__':
    args = constant.args
    print("="*50)
    print("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    print("TRAINING MANIFEST: ", args.train_manifest_list)
    print("="*50)

    if not os.path.exists("./log"):
        os.mkdir("./log")

    logging.basicConfig(filename="log/" + args.name, filemode='w+', format='%(asctime)s - %(message)s', level=logging.INFO)

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    logging.info(audio_conf)

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
    label2id, id2label = {}, {}
    count = 0
    for i in range(len(labels)):
        if labels[i] not in label2id:
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1
        else:
            print("multiple label: ", labels[i])

    train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.train_manifest_list, label2id=label2id, normalize=True, augment=args.augment)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = AudioDataLoader(
        train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

    start_epoch = 0
    metrics = None
    loaded_args = None
    if constant.args.continue_from != "":
        logging.info("Continue from checkpoint: " + constant.args.continue_from)
        print("Continue from checkpoint: " + constant.args.continue_from)
        model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(
            constant.args.continue_from)

        verbose = constant.args.verbose

        if loaded_args != None:
            # Unwrap nn.DataParallel
            if loaded_args.parallel:
                logging.info("unwrap from DataParallel")
                model = model.module

            # Parallelize the batch
            if args.parallel:
                model = nn.DataParallel(model, device_ids=args.device_ids)
    else:
        if constant.args.model == "TRFS":
            model = init_transformer_model(constant.args, label2id, id2label)
            opt = init_optimizer(constant.args, model, "noam")
        else:
            logging.info("The model is not supported, check args --h")
    
    loss_type = args.loss

    if constant.USE_CUDA:
        model = model.cuda(0)

    logging.info(model)
    num_epochs = constant.args.epochs

    trainer = TrainerVanilla()
    trainer.train(model, train_loader, train_sampler, opt, loss_type, start_epoch, num_epochs, label2id, id2label, metrics)