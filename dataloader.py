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

    def __init__(self, data_path, ann_path, vocab_size, max_cap_len, transform=None):

        self.coco = COCO(ann_path)
        self.ann_ids = self.coco.getAnnIds(imgIds=[])

        self.transform = transform
        self.path = data_path
        self.vocab_size = vocab_size
        self.max_cap_len = max_cap_len

        with open('word_to_id.p', 'rb') as fp:
            self.dictionary = pickle.load(fp)

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, ID):
        
        #return a coco anotation ID from the given ID
        ann_list = self.coco.loadAnns(ids=self.ann_ids[ID])
        
        # convert coco annotation ID to a caption and imageID
        caption = [ann['caption'] for ann in ann_list]
        image_id = [idx['image_id'] for idx in ann_list]

        # load images from disk
        img_path = self.coco.loadImgs(image_id)[0]['file_name']
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')

        # image transforms
        if self.transform:
            img = self.transform(img)

        # tokenize caption
        caption_tknID, word_mask = caption_to_id(caption, self.dictionary, self.vocab_size, self.max_cap_len)
        print(word_mask)
        print(caption_tknID)
        return img, caption, torch.LongTensor(caption_tknID), torch.IntTensor(word_mask)


# ================================
# Convert ID to Words
# ================================
def id_to_word(tkn_list, conversion_array):
    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]


# ================================
# Input a single caption (string), add start/end/unknown tokens, then convert to IDs
# Note: The dictionary must be first truncated to the vocab size
# ================================
def caption_to_id(caption, dictionary, vocab_size, max_cap_len):

    # tokenize caption
    caption_tkn = nltk.word_tokenize(caption[0])
    caption_tkn = [w.lower() for w in caption_tkn]

    # insert start / end tokens
    caption_tkn.insert(0, '<S>')
    caption_tkn.append('</S>')


    # initialize
    caption_tknID = [0] * (max_cap_len)
    word_mask = [0] * (max_cap_len)

    # insert unknown token
    for i, tkn in enumerate(caption_tkn):
        
        # if caption is longer than max caption length, break.
        # add 3 for the start/stop token and period.
        if i == (max_cap_len):
            break

        if (tkn not in dictionary):
            caption_tkn[i] = 'UNK'


        # truncate dictionary based on vocab_size        
        if caption_tknID[i] >= vocab_size:
            caption_tknID[i] = 3
        else:
            # lookup tokenID
            caption_tknID[i] = dictionary.get(tkn)

        # Update Word Mask
        word_mask[i] = 1



        # Special Tokens:
        # <S>: 1
        # </S>: 2
        # UNK: 3

    # convert word tokens to id
    return (caption_tknID, word_mask)





# ================================
# Load Data Function
# ================================
def load_data(path, batch_size, vocab_size, max_cap_len, n_workers=4):

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
        path, 'train2017/'), 
        ann_path=os.path.join(path, 'annotations/captions_train2017.json'),
        vocab_size = vocab_size,
        max_cap_len = max_cap_len,
        transform=data_transforms['train']
        )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    valset = coco_loader(data_path=os.path.join(
        path, 'val2017/'), 
        ann_path=os.path.join(path, 'annotations/captions_val2017.json'),
        vocab_size = vocab_size,
        max_cap_len = max_cap_len,
        transform=data_transforms['val']
        )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    return trainloader, valloader


