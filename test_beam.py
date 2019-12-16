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
from torchvision import models                                                                    
import sys
sys.path.insert(0, '../coco_data2014')

import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
from eval import eval_accy
import glob
import shutil
import os
os.chdir(os.path.dirname(os.path.realpath(__file__))) # needed for BlueWaters

sys.path.append('coco-caption')

from torchvision import models
import torch.nn as nn
import numpy as np
from datetime import datetime

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
    trainloader, data_loader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len, num_caps_per_img = args.num_caps_per_img)
    coco_testaccy = COCO(os.path.join(args.data_path, 'annotations/captions_val2014.json'))

    batchsize = args.batch_size
    max_tokens = args.max_cap_len
    num_batches = 100
    print('[DEBUG] Running test (w/ beam search) on %d batches' % num_batches)

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

    model_vgg.load_state_dict(checkpoint['state_dict_img'])
    
    model_imgcnn = model_vgg
    model_convcap = model_cc
    model_imgcnn.cuda()
    model_convcap.cuda()
    model_imgcnn.train(False) 
    model_convcap.train(False)
    with open('./embed/word_to_id.p', 'rb') as fp:
        dictionary = pickle.load(fp)

    data = np.load('./embed/id_to_word.npy')

    pred_captions = []
    for batch_idx, (imgs, _, _, img_ids) in \
        tqdm(enumerate(data_loader), total=num_batches):
        if batch_idx > 99:
            break
        imgs = imgs.view(batchsize, 3, 224, 224)

        imgs_v = Variable(imgs.cuda())
        imgsfeats, imgsfc7 = model_imgcnn(imgs_v)

        b, f_dim, f_h, f_w = imgsfeats.size()
        imgsfeats = imgsfeats.unsqueeze(1).expand(\
          b, args.beam_size, f_dim, f_h, f_w)
        imgsfeats = imgsfeats.contiguous().view(\
          b*args.beam_size, f_dim, f_h, f_w)

        beam_searcher = beamsearch(args.beam_size, batchsize, max_tokens)
  
        wordclass_feed = np.zeros((args.beam_size*batchsize, max_tokens), dtype='int64')
        wordclass_feed[:,0] = dictionary.get('<S>')
        imgsfc7 = repeat_img(args, imgsfc7)
        outcaps = np.empty((batchsize, 0)).tolist()

        for j in range(max_tokens-1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

            wordact, _ = model_convcap(wordclass, imgsfc7, imgsfeats)
            wordact = wordact[:,:,:-1]
            wordact_j = wordact[..., j]

            beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j) 

            if len(beam_indices) == 0 or j == (max_tokens-2): # Beam search is over.
                generated_captions = beam_searcher.get_results()
                for k in range(batchsize):
                    g = generated_captions[:, k]
                    outcaps[k] = [data[x] for x in g]
            else:
                wordclass_feed = wordclass_feed[beam_indices]
                imgsfc7 = imgsfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                imgsfeats = imgsfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j+1] = wordclass_idx

        for j in range(batchsize):
            num_words = len(outcaps[j]) 
            if '</S>' in outcaps[j]:
                num_words = outcaps[j].index('</S>')
            outcap = ' '.join(outcaps[j][:num_words])
            pred_captions.append({'image_id': img_ids[j], 'caption': outcap})

    scores = language_eval(pred_captions, args.model_dir, split)

    model_imgcnn.train(True) 
    model_convcap.train(True)

    return scores

def language_eval(input_data, savedir, split):
  if type(input_data) == str: # Filename given.
    checkpoint = json.load(open(input_data, 'r'))
    preds = checkpoint
  elif type(input_data) == list: # Direct predictions give.
    preds = input_data
  
  annFile = '../coco_data2014/annotations/captions_val2014.json'
  coco = COCO(annFile)
  valids = coco.getImgIds()

  # Filter results to only those in MSCOCO validation set (will be about a third)
  preds_filt = [p for p in preds if p['image_id'] in valids]
  len_p = len(preds_filt)
  for i in range(len_p):
    preds_filt[i]['image_id'] = int(preds_filt[i]['image_id'])
    pattern = str(preds_filt[i]['image_id'])
    '''
    #RUN THIS PORTION ONLY DURING INFERENCE ON FEW IMAGES
    for file in glob.glob(r'../coco_data2014/val2014/*'+pattern+'*'):
      shutil.copy(file, './saved_models/third/')
    '''
  print('Using %d/%d predictions' % (len(preds_filt), len(preds)))
  
  resFile1 = osp.join(savedir, 'result_%s.json' % (split))
  json.dump(preds_filt, open(resFile1, 'w')) # Serialize to temporary json file. Sigh, COCO API...
  
  
  
  resFile = json.dumps(preds_filt)
  cocoRes = coco.loadRes(resFile)
  cocoEval = COCOEvalCap(coco, cocoRes)
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  cocoEval.evaluate()

  # Create output dictionary.
  out = {}
  for metric, score in cocoEval.eval.items():
    out[metric] = score


  # Return aggregate and per image score.
  return out, cocoEval.evalImgs

