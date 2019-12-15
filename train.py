import os
os.chdir(os.path.dirname(os.path.realpath(__file__))) # needed for BlueWaters

import sys
sys.path.append('coco-caption')

import torch
from torchvision import models
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
import argparse
from datetime import datetime
import json

from models import conv_captioning
from dataloader import load_data
from eval import test_accy, id_to_word, gen_caption, dict_to_df
from img_encoders import vgg_extraction, resnet_extraction, densenet_extraction

# ======================================================
    # Input Parameters
# ======================================================

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default=os.path.dirname('../coco_data2014/'), help='path where data & annotations are located')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--vocab_size', type=int, default=9221)
parser.add_argument('--max_cap_len', type=int, default=15, help = 'maximum caption length')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default= 5e-5)
parser.add_argument('--scheduler_gamma', type=float, default=0.1)
parser.add_argument('--scheduler_stepsize', type=int, default=15)
parser.add_argument('--num_layers', type=int, default=3, help = 'number of convolution layers')
parser.add_argument('--kernel_size', type=int, default=5, help = 'size of 1d convolution kernel (how many words in the past?)')
parser.add_argument('--img_feat', type=int, default=512, help = 'number of features in image embedding layer. Should be divisible by 2.')
parser.add_argument('--word_feat', type=int, default=512, help = 'number of features in word embedding layer. Should be divisible by 2.')
parser.add_argument('--dropout_p', type=float, default=0.1, help = 'dropout probability parameter')
parser.add_argument('--train_vgg', type=int, default=8, help = 'the number of epochs after which the image extractor network will start training')
parser.add_argument('--attention', type=bool, default=False, help = 'use attention?')
parser.add_argument('--num_caps_per_img', type=int, default=5, help = 'number of captions per image in training set (should be 5 in coco)')
parser.add_argument('--model_save_path', type=str, default=os.path.dirname('./saved_models/'), help = 'where models are saved')
parser.add_argument('--load_model', type=str, default=None, help = 'provide the path of a model if you are loading a checkpoint')
parser.add_argument('--accy_file', type=str, default='./saved_models/model_accuracy.json', help='provide the accuracy results file if you are loading a checkpoint')
parser.add_argument('--temperature', type=float, default=1, help='temperature softmax')
parser.add_argument('--print_accy', type=int, default=1, help='how often to calculate test accy (# epochs)')
parser.add_argument('--img_model', type=str, default='vgg', help='vgg, resnet, or densenet')
parser.add_argument('--num_test_batches', type=int, default=500, help='number of batches to use when calculating test accy. always uses full dataset for the last epoch.')
parser.add_argument('--use_glove', type=bool, default=False, help='use the pre-trained glove features for the word embedding?')
parser.add_argument('--freeze_embed', type=bool, default=False, help='freeze the word embed layer')

args = parser.parse_args()

# ======================================================
    # Initialize Model
# ======================================================
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data
trainloader, valloader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len, num_caps_per_img = args.num_caps_per_img)
coco_testaccy = COCO(os.path.join(args.data_path, 'annotations/captions_val2014.json')) # create coco object for test accuracy calculation

# Initialize Models
model_cc = conv_captioning(args.vocab_size, args.kernel_size, args.num_layers, args.dropout_p, args.word_feat, args.img_feat + args.word_feat, args)
model_cc_params = filter(lambda p: p.requires_grad, model_cc.parameters())

model_cc.to(device)

if args.img_model == 'resnet':
    print('Loading Resnet18 Image Encoder...')
    model_vgg = resnet_extraction(args.img_feat)
elif args.img_model == 'densenet':
    print('Loading Densenet161 Image Encoder...')
    model_vgg = densenet_extraction(args.img_feat)
else:
    print('Loading VGG16 Image Encoder...')
    model_vgg = vgg_extraction(args.img_feat)
model_vgg.to(device)
print('Temperature = ', args.temperature)
print('Attention = ', args.attention)

# Initialize optimizer
optimizer = torch.optim.RMSprop(model_cc_params, lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )
optimizer_vgg = torch.optim.RMSprop(model_vgg.parameters(), lr = args.initial_lr)
scheduler_vgg = torch.optim.lr_scheduler.StepLR(optimizer_vgg, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

# Criterion
criterion = nn.CrossEntropyLoss()

test_scores = [] # initialize saved scores
init_epoch = 0

# ======================================================
    # Load Model
# ======================================================

if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    init_epoch = checkpoint['epoch'] + 1
    model_cc.load_state_dict(checkpoint['state_dict_cap'])
    optimizer.load_state_dict(checkpoint['optimizer_cap'])
    scheduler.load_state_dict(checkpoint['scheduler_cap'])

    model_vgg.load_state_dict(checkpoint['state_dict_img'])
    optimizer_vgg.load_state_dict(checkpoint['optimizer_img'])
    scheduler_vgg.load_state_dict(checkpoint['scheduler_img'])

    test_scores = json.load(open(args.accy_file, 'r'))



x = test_accy(valloader, coco_testaccy, model_vgg, model_cc, args) # calc test accuracy
for i in range(args.max_cap_len):
    print(len(np.unique(np.asarray(x[i]))))
print('End Training at Epoch ', args.num_epoch -1)