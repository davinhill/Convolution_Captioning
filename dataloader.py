import sys
sys.path.append('coco-caption')

import os
import torch
from torchvision import datasets, transforms
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle
import nltk
import numpy as np
nltk.download('punkt')


# I still need to fix the issue of caption length
# Also, need to figure out how to limit vocabulary size
# Shortest caption: 6. Longest caption: 57. Mean = 11.3, Med = 11 (ex. start/stop tokens)

# ================================
# Data Loader Class
# ================================
class coco_loader(Dataset):

    def __init__(self, data_path, ann_path, vocab_size, max_cap_len, transform=None, num_caps_per_img = 5):

        self.coco = COCO(ann_path)
        self.img_ids = self.coco.getImgIds(imgIds=[])

        self.transform = transform
        self.path = data_path
        self.vocab_size = vocab_size
        self.max_cap_len = max_cap_len

        with open('word_to_id.p', 'rb') as fp:
            self.dictionary = pickle.load(fp)
        
        self.num_captions_per_img = num_caps_per_img

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, ID):
        
        sample_image_id = self.img_ids[ID]

        # load images from disk
        img_path = self.coco.loadImgs(sample_image_id)[0]['file_name']
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')

        # image transforms
        if self.transform:
            img = self.transform(img)  # 3 x 224 x 224

        # get captions from image ID
        ann_ids = self.coco.getAnnIds(sample_image_id)
        cap_dict = self.coco.loadAnns(ann_ids)
        caption = [item['caption'] for item in cap_dict]  # list of 5 captions

        # tokenize caption and convert to IDs
        caption_tknID = []
        for i in range(self.num_captions_per_img):
            caption_tknID.append(caption_to_id(caption[i], self.dictionary, self.vocab_size, self.max_cap_len))

        return img, caption, torch.LongTensor(caption_tknID), sample_image_id
    





# ======================================================
    # Dataloader for the validation set (loads by imageID instead of annotationID)
# ======================================================
class coco_loader_val(Dataset):

    def __init__(self, data_path, ann_path, vocab_size, max_cap_len, transform=None):

        self.coco = COCO(ann_path)
        self.img_ids = self.coco.getImgIds(imgIds=[])

        self.transform = transform
        self.path = data_path


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, ID):

        # load images from disk
        img_path = self.coco.loadImgs(self.img_ids[ID])[0]['file_name']
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')

        # image transforms
        if self.transform:
            img = self.transform(img)
        '''
        ann_ids = self.coco.getAnnIds(self.img_ids[ID])
        cap_dict = self.coco.loadAnns(ann_ids)
        captions = [item['caption'] for item in cap_dict] 
        '''

        return img, self.img_ids[ID]

# ================================
# Input a single caption (string), add start/end/unknown tokens, then convert to IDs
# Note: The dictionary must be first truncated to the vocab size
# ================================
def caption_to_id(caption, dictionary, vocab_size, max_cap_len):

    # tokenize caption
    caption_tkn = nltk.word_tokenize(caption)
    caption_tkn = [w.lower() for w in caption_tkn]

    # insert start / end tokens
    caption_tkn.insert(0, '<S>')
    caption_tkn.append('</S>')


    # initialize
    caption_tknID = [0] * (max_cap_len)

    # insert unknown token
    for i, tkn in enumerate(caption_tkn):
        
        # if caption is longer than max caption length, break.
        # add 3 for the start/stop token and period.
        if i == (max_cap_len):
            break

        # words not in dictionary
        if (tkn not in dictionary):
            tkn = 'UNK'

        # lookup tokenID
        caption_tknID[i] = dictionary.get(tkn)

        # truncate dictionary based on vocab_size        
        if caption_tknID[i] >= vocab_size:
            caption_tknID[i] = 3

        # Special Tokens:
        # <S>: 1
        # </S>: 2
        # UNK: 3

    # convert word tokens to id
    return (caption_tknID)





# ================================
# Load Data Function
# ================================
def load_data(path, batch_size, vocab_size, max_cap_len, n_workers=4, num_caps_per_img=5):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
        ]),

        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
        ])
    }

    trainset = coco_loader(data_path=os.path.join(
        path, 'train2014/'), 
        ann_path=os.path.join(path, 'annotations/captions_train2014.json'),
        vocab_size = vocab_size,
        max_cap_len = max_cap_len,
        transform=data_transforms['train'],
        num_caps_per_img = num_caps_per_img,
        )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    valset = coco_loader_val(data_path=os.path.join(
        path, 'val2014/'), 
        ann_path=os.path.join(path, 'annotations/captions_val2014.json'),
        vocab_size = vocab_size,
        max_cap_len = max_cap_len,
        transform=data_transforms['val']
        )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    return trainloader, valloader


