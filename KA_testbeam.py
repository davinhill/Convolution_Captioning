# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:05:01 2019

@author: krish
"""
import torch 
beam_size=3

device=torch.device('cuda' if torch.cuda.is_available() else'cpu')


def repeat_img(args, img_emb):

    batch_size=img_emb.size(0)
    
    feat_dim = img_emb.size(1)
    
    assert img_emb.dim() == 2
    
    img_emb_new = img_emb.data.new(batch_size*args.beam_size,\
                                   feat_dim).to(device)
    
    #Here I assume that the img_emb_new is a Tensor instead of wrapping it into a variable
    
    for i in range(batch_size):
        start_ind = i * args.beam_size
        img_emb_new[start_ind:start_ind+args.beam_size, :] = img_emb[i, :].repeat(args.beam_size, 1)
 
    return img_emb_new
