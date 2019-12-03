import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.append('coco-caption')

import torch
from torchvision import models
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO

import argparse
from datetime import datetime

from models import conv_captioning, vgg_extraction
from dataloader import load_data
from eval import test_accy

# ======================================================
    # Input Parameters
# ======================================================

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default=os.path.dirname('../coco_data2017/'), help='path where data & annotations are located')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--vocab_size', type=int, default=9221)
parser.add_argument('--max_cap_len', type=int, default=15, help = 'maximum caption length')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default=5 * np.exp(-5))
parser.add_argument('--scheduler_gamma', type=float, default=0.1)
parser.add_argument('--scheduler_stepsize', type=int, default=15)
parser.add_argument('--num_layers', type=int, default=3, help = 'number of convolution layers')
parser.add_argument('--kernel_size', type=int, default=5, help = 'size of 1d convolution kernel (how many words in the past?)')
parser.add_argument('--img_feat', type=int, default=512, help = 'number of features in image embedding layer. Should be divisible by 2.')
parser.add_argument('--word_feat', type=int, default=512, help = 'number of features in word embedding layer. Should be divisible by 2.')
parser.add_argument('--dropout_p', type=float, default=0.1, help = 'dropout probability parameter')
parser.add_argument('--train_vgg', type=int, default=8, help = 'the number of epochs after which the image extractor network will start training')
parser.add_argument('--attention', type=bool, default=False, help = 'use attention?')

args = parser.parse_args()

# ======================================================
    # Initialize Model
# ======================================================
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data
trainloader, valloader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len)
coco_testaccy = COCO(os.path.join(args.data_path, 'annotations/captions_val2017.json')) # create coco object for test accuracy calculation

# Initialize Models
model_vgg = vgg_extraction(args.img_feat)
model_vgg.to(device)
model_cc = conv_captioning(args.vocab_size, args.kernel_size, args.num_layers, args.dropout_p, args.word_feat, args.img_feat + args.word_feat)
model_cc.to(device)

# Initialize optimizer
optimizer = torch.optim.RMSprop(model_cc.parameters(), lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

# Criterion
criterion = nn.CrossEntropyLoss()

train_vgg = False  # initialize flag so that the vgg network is not trained at start of training

# ======================================================
    # Train
# ======================================================

for epoch in range(args.num_epochs):
    import pdb; pdb.set_trace()
    epoch_time_start = datetime.now()

    # start training vgg after specified number of epochs
    if epoch == args.train_vgg:
        train_vgg = True
        optimizer_vgg = torch.optim.RMSprop(model_vgg.parameters(), lr = args.initial_lr)
        scheduler_vgg = torch.optim.lr_scheduler.StepLR(optimizer_vgg, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

    model_cc.train()
    if train_vgg:
        model_vgg.train()
    else:
        model_vgg.eval() # will putting it in eval mode impact caption training?


    for batchID, (image, caption, caption_tknID, imgID) in enumerate(trainloader):
        batch_start = datetime.now() 
        optimizer.zero_grad()
        if train_vgg:
            optimizer_vgg.zero_grad()
        image, caption_tknID = image.to(device), caption_tknID.to(device)

        # run models
        img_conv, img_fc = model_vgg(image) # extract image features

        if args.attention:
            placeholder = 0
        else:
            pred = model_cc(caption_tknID, img_fc)  # generate predicted caption. n x vocab_size x max_cap_len

        # reshape predicted and GT captions for loss calculation
        batch_size = pred.shape[0] # get current batch size
        caption_pred = pred.transpose(1, 2).reshape(batch_size * args.max_cap_len, -1) # n * max_cap_len x vocab_size (probability dist'n over all words)
        caption_target = caption_tknID.reshape(batch_size * args.max_cap_len)  # n * max_cap_len x 1
        word_mask = caption_target.nonzero().reshape(-1) # the word mask filters out "unused words" when the GT caption is shorter than the max caption length.

        # calculate Cross-Entropy loss
        loss = criterion(caption_pred[word_mask, :], caption_target[word_mask])   

        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_vgg:
            optimizer_vgg.step()
            scheduler_vgg.step()

        if batchID % 100 == 0:
            epoch_time = datetime.now() - batch_start
            print("Batch: %d || Loss: %f || Time: %s" % (batchID, loss, str(epoch_time)))

    
    accy = test_accy(valloader, coco_testaccy, model_vgg, model_cc, args.max_cap_len)
    epoch_time = datetime.now() - epoch_time_start
    print("========================================")
    print("Epoch: %d || Loss: %f || Time: %s" % (epoch, loss, str(epoch_time)))
    print(accy['Bleu_1'])
    print(accy['Bleu_2'])
    print(accy['Bleu_3'])
    print(accy['Bleu_4'])
    print("========================================")

