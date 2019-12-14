import argparse
import numpy as np 
import time 
import pickle 
import itertools
import os.path as osp


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 

from beamsearch import beamsearch 
from coco_loader import coco_loader
from torchvision import models                                                                    
#from convcap import convcap
#from vggfeats import Vgg16Feats
from evaluate import language_eval

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

def repeat_img(args, img_emb):
    b = img_emb.size(0) #10
    assert img_emb.dim() == 2

    img_emb_new = Variable(img_emb.data.new(img_emb.size(0)*args.beam_size, img_emb.size(1))).cuda() #30*512
    for k in range(b):
        start_idx = k * args.beam_size
        img_emb_new[start_idx:start_idx+args.beam_size, :] = img_emb[k, :].repeat(args.beam_size, 1)
    return img_emb_new

def test_beam(args, split, modelfn=None): 
    """Sample generation with beam-search"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    #data = coco_loader(args.coco_root, split=split, ncap_per_img=1)
    #print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

    #data_loader = DataLoader(dataset=data, num_workers=args.nthreads,\
    #batch_size=args.batchsize, shuffle=False, drop_last=True)
    #import json
    #with open('captions_val2014.json', 'r') as f:
    #    data = json.load(f)
    #    data['type'] = 'captions'
    #with open('captions_val2014.json', 'w') as f:
    #    json.dump(data, f)

    #with open('captions_train2014.json', 'r') as f:
    #    data = json.load(f)
    #    data['type'] = 'captions'
    #with open('captions_train2014.json', 'w') as f:
    #    json.dump(data, f)
    trainloader, data_loader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len, num_caps_per_img = args.num_caps_per_img)
    coco_testaccy = COCO(os.path.join(args.data_path, 'annotations/captions_val2014.json'))

    batchsize = args.batch_size #say 10
    max_tokens = args.max_cap_len #15
    num_batches = 5
    print('[DEBUG] Running test (w/ beam search) on %d batches' % num_batches)
  
    #model_imgcnn = Vgg16Feats()
    #model_imgcnn.cuda() 

    #model_convcap = convcap(data.numwords, args.num_layers, is_attention=args.attention)
    #model_convcap.cuda()

    #print('[DEBUG] Loading checkpoint %s' % modelfn)
    #checkpoint = torch.load(modelfn)
    #model_convcap.load_state_dict(checkpoint['state_dict'])
    #model_imgcnn.load_state_dict(checkpoint['img_state_dict'])
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

    checkpoint = torch.load(modelfn)
    model_cc.load_state_dict(checkpoint['state_dict_cap'])
    #optimizer.load_state_dict(checkpoint['optimizer_cap'])
    #scheduler.load_state_dict(checkpoint['scheduler_cap'])

    model_vgg.load_state_dict(checkpoint['state_dict_img'])
    #optimizer_vgg.load_state_dict(checkpoint['optimizer_img'])
    #scheduler_vgg.load_state_dict(checkpoint['scheduler_img'])
    
    model_imgcnn = model_vgg
    model_convcap = model_cc
    model_imgcnn.cuda()
    model_convcap.cuda()
    model_imgcnn.train(False) 
    model_convcap.train(False)
    with open('./embed/word_to_id.p', 'rb') as fp:
        dictionary = pickle.load(fp)
    #with open('./embed/id_to_word.p', 'rb') as fp2:
    #    dictionary2 = pickle.load(fp2)
    #print(type(dictionary))
    #print(dictionary.keys())
    data = np.load('./embed/id_to_word.npy')
    print(type(data))
    print(data.shape)
    #print(dictionary.values())
    #print(dictionary.get('raced'))
    #print(dictionary.get('<S>'))
    pred_captions = []
    for batch_idx, (imgs, _, _, img_ids) in \
        tqdm(enumerate(data_loader), total=num_batches):
        if batch_idx > 4:
            break
        imgs = imgs.view(batchsize, 3, 224, 224)

        imgs_v = Variable(imgs.cuda())
        imgsfeats, imgsfc7 = model_imgcnn(imgs_v) #imgsfeats has size 10*a*b*c | imgsfc7 has size 10*512

        b, f_dim, f_h, f_w = imgsfeats.size()
        imgsfeats = imgsfeats.unsqueeze(1).expand(\
          b, args.beam_size, f_dim, f_h, f_w) #10*3*a*b*c
        imgsfeats = imgsfeats.contiguous().view(\
          b*args.beam_size, f_dim, f_h, f_w) #30*a*b*c

        beam_searcher = beamsearch(args.beam_size, batchsize, max_tokens)
  
        wordclass_feed = np.zeros((args.beam_size*batchsize, max_tokens), dtype='int64') #30*15
        wordclass_feed[:,0] = dictionary.get('<S>') #data.wordlist.index('<S>') #basically it becomes 30*15*9000
        imgsfc7 = repeat_img(args, imgsfc7) #30*512 --> 3 copies of 10*512
        outcaps = np.empty((batchsize, 0)).tolist() #[[], [], [], [], [], [], [], [], [], []]

        for j in range(max_tokens-1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda() #30*15*9000

            wordact, _ = model_convcap(wordclass, imgsfc7, imgsfeats) #wordact - 30 x 9000 x 15
            wordact = wordact[:,:,:-1] #wordact - 30 x 9000 x 14
            wordact_j = wordact[..., j] #wordact - 30 x 9000

            beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j) 

            if len(beam_indices) == 0 or j == (max_tokens-2): # Beam search is over.
                generated_captions = beam_searcher.get_results()
                for k in range(batchsize):
                    g = generated_captions[:, k]
                    #print(g)
                    outcaps[k] = [data[x] for x in g]
            else:
                wordclass_feed = wordclass_feed[beam_indices]
                imgsfc7 = imgsfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                imgsfeats = imgsfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j+1] = wordclass_idx

        for j in range(batchsize):
            num_words = len(outcaps[j]) 
            if 'EOS' in outcaps[j]:
                num_words = outcaps[j].index('EOS')
            outcap = ' '.join(outcaps[j][:num_words])
            pred_captions.append({'image_id': img_ids[j], 'caption': outcap})

    scores = language_eval(pred_captions, args.model_dir, split)

    model_imgcnn.train(True) 
    model_convcap.train(True)

    return scores
