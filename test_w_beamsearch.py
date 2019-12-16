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
from eval import test_accy, id_to_word, gen_caption
from img_encoders import vgg_extraction, resnet_extraction, densenet_extraction
from test_beam import test_beam
import os.path as osp
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
parser.add_argument('--beam_size', type=int, default=1, help='beam size')
parser.add_argument('--use_glove', type=bool, default=False, help='use the pre-trained glove features for the word embedding?')
parser.add_argument('--freeze_embed', type=bool, default=False, help='freeze the word embed layer')
parser.add_argument('--coco_root', type=str, default= './data/coco/', help='directory containing coco dataset train2014, val2014, & annotations')
parser.add_argument('--model_dir', type=str, default=os.path.dirname('./saved_models/'), help = 'where models are saved')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================================================
    # Load Best Model
# ======================================================
        
bestmodelfn = osp.join(args.load_model, 'best_model.pt')

if(args.beam_size > 1):
    scores = test_beam(args, 'val', modelfn=bestmodelfn)
else:
    raise Exception('No checkpoint found %s' % bestmodelfn)
