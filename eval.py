import sys
sys.path.append('/coco-caption')

import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
import json
import torch.nn.functional as F
import torch


def train_accy(args):

    #coco_trainaccy = COCOEvalCap(cocoGT = COCO(os.path.join(args.data_path, 'annotations/captions_val2017.json')))
    prob_input = prob_input.cpu().detach()
    wordprob = F.softmax(prob_input, dim = 1)
    tknID = np.argmax(wordprob, axis = 1)

    # convert IDs to words
    id_conversion_array = np.load('id_to_word.npy')
    caption_output = []
    for i in range(tknID.shape[0]):
        caption_output.append(id_to_word(tknID[i, :], id_conversion_array))

# ================================
# Convert ID to Words
# ================================
def id_to_word(tkn_list, conversion_array):
    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]


def eval_accy():
    return 0


# ================================
# Generate a caption for a given image based on the trained model
# image input should be of shape n x 3 x 224 x 224
# ================================
def gen_caption(image, image_model, caption_model, max_cap_len = 15):

    batch_size = image.shape[0]
    caption_tknID = torch.zeros(batch_size, max_cap_len, dtype = torch.long)# initialize tkn predictions
    caption_tknID[:,0] = 1   # <S> token

    # Set models to eval mode and move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_model.to(device).eval()
    caption_model.to(device).eval()
    image, caption_tknID = image.to(device), caption_tknID.to(device)

    # get image features
    img_conv, img_fc = image_model(image)  

    for i in range(max_cap_len-1):

        # generate model predictions for the next word, based on the previously-"stored" predictions
        pred = caption_model(caption_tknID, img_fc).cpu().detach()  # n x vocab_size x max_cap_len
        pred = np.argmax(pred, axis = 1)    # n x max_cap_len

        # update "stored" predictions
        caption_tknID[:, i+1] = pred[:, i+1]

    
    # convert IDs to words
    id_conversion_array = np.load('id_to_word.npy')
    caption_tkn = []
    for i in range(batch_size):
        caption_tkn.append(id_to_word(caption_tknID[i, :], id_conversion_array))


    # convert word lists to strings
    caption_str = []
    for i in range(batch_size):
        if '</S>' in caption_tkn[i]:
            caption_str.append(' '.join(caption_tkn[i][1:caption_tkn.index('</S>')]))
        else:
            caption_str.append(' '.join(caption_tkn[i][1:]))
            
    return caption_str


# ================================
# Convert a tokenized caption represented by ID#s to string
# looks up ID values and removes start/end tokens
# input should be of dimension n x vocab_size x max_cap_len
# ================================
def test_accy(dataloader, caption_model):
    # should i have a different dataloader for validation that does not tokenize the caption?
    for batchID, (image, caption, caption_tknID) in enumerate(dataloader):
        print(batchID)
    return 0






