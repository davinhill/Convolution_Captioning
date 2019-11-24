import torch
from torchvision import models
import torch.nn as nn
import numpy as np

import argparse
import os
from datetime import datetime

from models import conv_captioning, vgg_extraction
from dataloader import load_data, id_to_word

# ======================================================
    # Input Parameters
# ======================================================

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='path where data & annotations are located')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--vocab_size', type=int, default=9221)
parser.add_argument('--max_cap_len', type=int, default=15, help = 'maximum caption length')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_lr', type=int, default=5 * np.exp(-5))
parser.add_argument('--scheduler_gamma', type=int, default=0.1)
parser.add_argument('--scheduler_stepsize', type=int, default=15)
parser.add_argument('--num_layers', type=int, default=3, help = 'number of convolution layers')
parser.add_argument('--kernel_size', type=int, default=5, help = 'size of 1d convolution kernel')
parser.add_argument('--img_feat', type=int, default=512, help = 'number of features in image embedding layer. Should be divisible by 2.')
parser.add_argument('--word_feat', type=int, default=512, help = 'number of features in word embedding layer. Should be divisible by 2.')
parser.add_argument('--dropout_p', type=int, default=0.1, help = 'dropout probability parameter')

args = parser.parse_args()

# ======================================================
    # Initialize Model
# ======================================================
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader, valloader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len)


model_vgg = vgg_extraction(args.img_feat)
model_vgg.to(device)

model_cc = conv_captioning(args.vocab_size, args.kernel_size, args.num_layers, args.dropout_p, args.word_feat, args.img_feat + args.word_feat)
model_cc.to(device)



# Initialize optimizer
optimizer = torch.optim.RMSprop(model_cc.parameters(), lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

# Criterion
criterion = nn.CrossEntropyLoss()



# ======================================================
    # Train
# ======================================================

for epoch in range(args.num_epochs):

    epoch_time_start = datetime.now()

    for batchID, (image, _, caption_tknID, word_mask) in enumerate(valloader):
        batch_start = datetime.now() 
        optimizer.zero_grad()
        image, caption_tknID, word_mask = image.to(device), caption_tknID.to(device), word_mask.to(device)

        img_conv, img_fc = model_vgg(image)
        pred = model_cc(caption_tknID, img_fc)  # n x vocab_size x max_cap_len

        caption_pred = pred.transpose(1, 2).reshape(args.batch_size * args.max_cap_len, -1) # n * max_cap_len x vocab_size (probability dist'n over all words)
        caption_target = caption_tknID.reshape(args.batch_size * args.max_cap_len, -1)  # n * max_cap_len x 1
        import pdb; pdb.set_trace()
        word_mask = word_mask.reshape(args.batch_size * args.max_cap_len, -1).nonzero()[:,0]

        loss = criterion(caption_pred[word_mask, :], caption_target[word_mask, 0])

        loss.backward()
        optimizer.step()

        epoch_time = datetime.now() - batch_start
        print("Epoch: %d || Loss: %f || Time: %s" % (epoch, loss, str(epoch_time)))

    
    epoch_time = datetime.now() - epoch_time_start
    print("Epoch: %d || Loss: %f || Time: %s" % (epoch, loss, str(epoch_time)))
