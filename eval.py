import os
os.chdir(os.path.dirname(os.path.realpath(__file__))) # needed for BlueWaters

import sys
sys.path.append('/coco-caption')

import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
import json
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd

# ================================
# Convert ID to Words
# tkn_list should be a single list of tknIDs. Returns a list of words.
# ================================
def id_to_word(tkn_list, conversion_array):
    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]

# ================================
# convert list of tokenized words to string
# input is a single list of tokenized words
# ================================
def wordlist_to_string(caption_tkn):

    # reduce string length if end token is present
    if '</S>' in caption_tkn:
        output = ' '.join(caption_tkn[1:caption_tkn.index('</S>')])
    else:
        output = ' '.join(caption_tkn[1:])
    
    return output


# ================================
# convert model_accuracy file to pandas dataframe and save
# ================================
def dict_to_df(accy_file, path):
    df = pd.DataFrame()

    for key in accy_file[0]:
        df[key] = [value[key] for value in accy_file]

    df.to_csv(os.path.join(path, 'model_accuracy.csv'))


# ================================
# evaluate accuracy metrics given predictions (uses CocoEvalAPI)
# 'predictions' is a list of dictionary objects, with 'image_id' and 'caption'
# ================================
def eval_accy(predictions, coco_object):
    resfile = json.dumps(predictions)
    coco_results = coco_object.loadRes(resfile)

    coco_eval = COCOEvalCap(coco_object, coco_results)
    coco_eval.params['image_id'] = coco_results.getImgIds()
    coco_eval.evaluate()

    output = {}
    for metric, score in coco_eval.eval.items():
        output[metric] = score

    return output


# ================================
# Generate a caption for a given image based on the trained model
# image input should be of shape n x 3 x 224 x 224
# if imgID (the list of image IDs assocated with the provided images), this function will return a list of dictionary
# objects, with 'image_id' and 'caption', for use with the CocoEvalAPI.
# ================================
def gen_caption(image, image_model, caption_model, vocab_size, max_cap_len = 15, imgID = None):

    batch_size = image.shape[0]
    caption_tknID = torch.zeros(batch_size, max_cap_len, dtype = torch.long)# initialize tkn predictions
    caption_tknID[:,0] = 1   # <S> token
    caption_prob = torch.zeros(batch_size, max_cap_len, vocab_size)

    # Set models to eval mode and move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_model.to(device).eval()
    caption_model.to(device).eval()
    image, caption_tknID = image.to(device), caption_tknID.to(device)

    # get image features
    img_conv, img_fc = image_model(image)  

    for i in range(max_cap_len-1):

        # generate model predictions for the next word, based on the previously-"stored" predictions
        pred, _ = caption_model(caption_tknID, img_fc, img_conv) # n x vocab_size x max_cap_len
        pred = pred.cpu().detach()
        pred_word = np.argmax(pred, axis = 1)    # n x max_cap_len

        # update "stored" predictions
        caption_tknID[:, i+1] = pred_word[:, i]
        caption_prob[:, i+1, :] = pred.transpose(1, 2)[:, i, :]
    
    # convert IDs to words
    id_conversion_array = np.load('./embed/id_to_word.npy')
    caption_tkn = []
    for i in range(batch_size):
        caption_tkn.append(id_to_word(caption_tknID[i, :], id_conversion_array))


    # convert word lists to strings
    caption_str = []
    for i in range(batch_size):

        # convert tokenized words to string
        output = wordlist_to_string(caption_tkn[i])

        # either append image ID (dict) or output only captions
        if (imgID is not None):
            caption_str.append({'image_id': imgID[i].item(), 'caption': output})
        else:
            caption_str.append(output)

    return caption_str, caption_tknID, caption_prob


# ================================
# Calculate test accuracy based on a given dataloader
# ================================
def test_accy(dataloader, coco_object, image_model, caption_model, epoch, args):
    with torch.no_grad():

        # initialize counters
        pred = []
        word_accy = 0
        loss = 0
        counter_num_words = 0
        counter_batch = 0
        criterion = nn.CrossEntropyLoss()

        # set number of batches on which to calculate test metrics
        for batchID, (image, _, caption_tknID, imgID) in enumerate(dataloader):
            caption_tknID = caption_tknID.squeeze()
            pred_caption_str, pred_caption_tknID, pred_caption_prob = gen_caption(image, image_model, caption_model, args.vocab_size, args.max_cap_len, imgID)
            pred.extend(pred_caption_str)
            # reshape caption to account for num captions per image
            batch_size = image.shape[0]

            # reshape pred / GT such that pred does not include <S>
            caption_tknID = caption_tknID[:, 1:] # (batch_size * 5) x (max_cap_len - 1)
            pred_caption_tknID = pred_caption_tknID[:, 1:] # (batch_size * 5) x (max_cap_len - 1)
            pred_caption_prob = pred_caption_prob[:, 1:, :]  # batch_size x (max_cap_len - 1) x vocab_size 

            caption_tknID = caption_tknID.flatten()
            pred_caption_tknID = pred_caption_tknID.flatten()
            pred_caption_prob = pred_caption_prob.flatten(end_dim = 1)
            word_mask = caption_tknID.nonzero().flatten() # the word mask filters out "unused words" when the GT caption is shorter than the max caption length.

            # calculate test loss 
            loss += criterion(pred_caption_prob[word_mask, :], caption_tknID[word_mask]).item()
            word_accy += sum(pred_caption_tknID[word_mask].cpu() == caption_tknID[word_mask].cpu()).item()
            counter_num_words += len(word_mask)
            counter_batch += 1

    return eval_accy(pred, coco_object), loss / counter_batch, word_accy / counter_num_words





