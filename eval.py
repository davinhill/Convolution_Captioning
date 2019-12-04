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
# evaluate accuracy metrics given predictions (uses CocoEvalAPI)
# 'predictions' is a list of dictionary objects, with 'image_id' and 'caption'
# ================================
def eval_accy(predictions, coco_object):
    resfile = 'tmp_resfile.json'
    json.dump(predictions, open(resfile, 'w'))
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
def gen_caption(image, image_model, caption_model, max_cap_len = 15, imgID = None):

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
    import pdb; pdb.set_trace()
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

        # convert tokenized words to string
        output = wordlist_to_string(caption_tkn[i])

        # either append image ID (dict) or output only captions
        if (imgID is not None):
            caption_str.append({'image_id': imgID[i].item(), 'caption': output})
        else:
            caption_str.append(output)

    return caption_str


# ================================
# Calculate test accuracy based on a given dataloader
# ================================
def test_accy(dataloader, coco_object, image_model, caption_model, max_cap_len):
    with torch.no_grad():
        pred = []
        for batchID, (image, image_id) in enumerate(dataloader):
            pred.extend(gen_caption(image, image_model, caption_model, max_cap_len, image_id))
    print(pred[0])
    print(pred[1])
    return eval_accy(pred, coco_object)





