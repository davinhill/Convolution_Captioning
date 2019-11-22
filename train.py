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
parser.add_argument('--vocab_size', type=int, default=10000)
parser.add_argument('--max_cap_len', type=int, default=15, help = 'maximum caption length')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_lr', type=int, default=5 * np.exp(-5))
parser.add_argument('--scheduler_gamma', type=int, default=0.1)
parser.add_argument('--scheduler_stepsize', type=int, default=15)

args = parser.parse_args()

# ======================================================
    # Initialize Model
# ======================================================
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader, valloader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len)

iterator = iter(valloader)

image, caption, caption_tknID, word_mask = next(iterator)

# Check that the tokenized caption is correct:
id_to_word_array = np.load('id_to_word.npy')
a = id_to_word(caption_tknID, id_to_word_array)

import pdb; pdb.set_trace()
model_vgg = vgg_extraction()
model_vgg.to(device)

model_cc = conv_captioning()
model_cc.to(device)



# Initialize optimizer
optimizer = torch.optim.RMSProp(model_cc.parameters(), lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

# Criterion
criterion = nn.CrossEntropyLoss()



# ======================================================
    # Train
# ======================================================

