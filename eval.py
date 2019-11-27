import numpy as np
from pycocotools.coco import COCO, COCOeval
import os
import json
import torch.nn.functional as F

def train_accy(args):

    coco_trainaccy = COCOeval(cocoGT = COCO(os.path.join(args.data_path, 'annotations/captions_val2017.json')))


# ================================
# Convert ID to Words
# ================================
def id_to_word(tkn_list, conversion_array):
    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]


# ================================
# Convert a numpy array of word probabilities (pre-softmax) to tokenIDs
# input should be of dimension n x vocab_size x max_cap_len
# ================================
def wordprob_to_tknCaption(prob_input, args):
    return 0




# ================================
# Convert a tokenized caption represented by ID#s to string
# looks up ID values and removes start/end tokens
# input should be of dimension n x vocab_size x max_cap_len
# ================================
def wordprob_to_string(prob_input, args):
    import pdb; pdb.set_trace()
    prob_input.cpu()
    wordprob = F.softmax(prob_input, dim = 1)
    tknID = np.argmax(wordprob, axis = 1)
    tknID.squeeze(1)

    id_conversion_array = np.load('id_to_word.npy')

    caption_output = []
    for i in range(tknID.shape[0]):
        caption_output.append(id_to_word(tknID[i, :], id_conversion_array))



class train_accy():
    def __init__(self):
        super(train_accy).__init__()

    coco_trainaccy = COCOeval(cocoGT = COCO(os.path.join(args.data_path, 'annotations/captions_val2017.json')))