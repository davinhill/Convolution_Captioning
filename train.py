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
from eval import test_accy, id_to_word, gen_caption, dict_to_df
from img_encoders import vgg_extraction, resnet_extraction, densenet_extraction

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
parser.add_argument('--use_glove', type=bool, default=False, help='use the pre-trained glove features for the word embedding?')
parser.add_argument('--freeze_embed', type=bool, default=False, help='freeze the word embed layer')

args = parser.parse_args()

# ======================================================
    # Initialize Model
# ======================================================
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data
trainloader, valloader = load_data(path = args.data_path, batch_size = args.batch_size, vocab_size = args.vocab_size, max_cap_len=args.max_cap_len, num_caps_per_img = args.num_caps_per_img)
coco_testaccy = COCO(os.path.join(args.data_path, 'annotations/captions_val2014.json')) # create coco object for test accuracy calculation

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

# Initialize optimizer
optimizer = torch.optim.RMSprop(model_cc_params, lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )
optimizer_vgg = torch.optim.RMSprop(model_vgg.parameters(), lr = args.initial_lr)
scheduler_vgg = torch.optim.lr_scheduler.StepLR(optimizer_vgg, gamma = args.scheduler_gamma, step_size = args.scheduler_stepsize )

# Criterion
criterion = nn.CrossEntropyLoss()

test_scores = [] # initialize saved scores
init_epoch = 0

# ======================================================
    # Load Model
# ======================================================

if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    init_epoch = checkpoint['epoch'] + 1
    model_cc.load_state_dict(checkpoint['state_dict_cap'])
    optimizer.load_state_dict(checkpoint['optimizer_cap'])
    scheduler.load_state_dict(checkpoint['scheduler_cap'])

    model_vgg.load_state_dict(checkpoint['state_dict_img'])
    optimizer_vgg.load_state_dict(checkpoint['optimizer_img'])
    scheduler_vgg.load_state_dict(checkpoint['scheduler_img'])

    test_scores = json.load(open(args.accy_file, 'r'))

# ======================================================
    # Train
# ======================================================

for epoch in range(init_epoch, args.num_epochs):

    epoch_time_start = datetime.now()
    counter_batch = 0
    counter_words = 0
    epoch_loss = 0
    epoch_word_accy = 0

    # start training vgg after specified number of epochs
    if epoch >= args.train_vgg:
        train_vgg = True
    else:
        train_vgg = False

    model_cc.train()
    model_vgg.train()


    for batchID, (image, caption, caption_tknID, imgID) in enumerate(trainloader):

        batch_start = datetime.now() 
        optimizer.zero_grad()
        if train_vgg:
            optimizer_vgg.zero_grad()
        image, caption_tknID = image.to(device), caption_tknID.to(device)
        #image.shape: batch_size x 3 x 224 x 224
        #caption_tknID.shape: batch_size x 5 x max_cap_len

        # reshape caption to account for num captions per image
        batch_size = image.shape[0] 
        caption_tknID = caption_tknID.reshape(batch_size * args.num_caps_per_img, args.max_cap_len) #batch_size * 5 x max_cap_len

        # run image feature extraction model
        img_conv, img_fc = model_vgg(image) # extract image features
        #img_conv.shape: batch_size x 512 x 7 x 7, img_fc.shape: batch_size x 512

        # repeat image features for the number of captions (5)
        img_fc = img_fc.unsqueeze(1).expand(-1, args.num_caps_per_img, -1) # batch_size x 5 x 512
        img_fc = img_fc.reshape(batch_size * args.num_caps_per_img, -1) # (batch_size * 5) x 512
        img_conv = img_conv.unsqueeze(1).expand(-1, args.num_caps_per_img, -1, -1, -1)
        img_conv = img_conv.reshape(batch_size * args.num_caps_per_img, 512, 7, 7)

        pred, attn_score = model_cc(caption_tknID, img_fc, img_conv)  # generate predicted caption. 
        # pred: n x vocab_size x max_cap_len
        # attn_score: n x max_cap_len x 49

        # reshape pred / GT such that pred does not include <S>
        pred = pred[:, :, :-1]  # batch_size x vocab_size x (max_cap_len - 1)
        caption_tknID = caption_tknID[:, 1:] # (batch_size * 5) x (max_cap_len - 1)

        # reshape predicted and GT captions for loss calculation (flatten)
        batch_size = batch_size * args.num_caps_per_img # new batch size after repeating image features per caption
        caption_pred = pred.transpose(1, 2).reshape(batch_size * (args.max_cap_len-1), -1) # n * (max_cap_len-1) x vocab_size (probability dist'n over all words)
        caption_target = caption_tknID.reshape(batch_size * (args.max_cap_len-1))  # n * (max_cap_len-1) x 1
        word_mask = caption_target.nonzero().reshape(-1) # the word mask filters out "unused words" when the GT caption is shorter than the max caption length.

        # calculate Cross-Entropy loss
        if args.attention:
            loss = criterion(caption_pred[word_mask, :]/args.temperature, caption_target[word_mask]) + \
                torch.mean(torch.pow(1 - torch.sum(attn_score, 1), 2))
        else:
            loss = criterion(caption_pred[word_mask, :]/args.temperature, caption_target[word_mask])   
        
        word_accy = sum(np.argmax(caption_pred[word_mask, :].cpu().detach(), axis = 1) == caption_target[word_mask].cpu())

        loss.backward()
        optimizer.step()
        if train_vgg:
            optimizer_vgg.step()

        # calculate avg epoch loss / accy
        epoch_loss += loss.item()
        epoch_word_accy += word_accy.item()
        counter_batch += 1
        counter_words += len(word_mask)

        if batchID % 3000 == 0:
            epoch_time = datetime.now() - batch_start
            print("Batch: %d || Loss: %f || Time: %s" % (batchID, loss, str(epoch_time)))
            
            # Print an example caption
            z, _, _ = gen_caption(image, model_vgg, model_cc, args.vocab_size, args.max_cap_len)
            print('TEST------------------')
            print(z[0])

            id_conversion_array = np.load('id_to_word.npy')
            y = caption_pred[:14, :].cpu().detach().numpy()
            y = torch.from_numpy(np.argmax(y, axis = 1).reshape(-1))
            y = id_to_word(y, id_conversion_array)
            print('TRAIN------------------')
            print(y)

            x = id_to_word(caption_target[:14], id_conversion_array)
            print('GT------------------------')
            print(x)
            print("=============================================")
    
    scheduler.step()
    if train_vgg:
        scheduler_vgg.step()

    epoch_time = datetime.now() - epoch_time_start
    epoch_loss /= counter_batch
    epoch_word_accy /= counter_words
    print("========================================")
    print("Epoch: %d || Train_Loss: %f || Train_Accy: %f || Time: %s" % (epoch, epoch_loss, epoch_word_accy, str(epoch_time)))
    print("========================================")
    if (epoch % args.print_accy == 0 and args.print_accy != 0) or (epoch == (args.num_epochs -1)):
        accy_time_start = datetime.now()
        accy, test_loss, test_waccy = test_accy(valloader, coco_testaccy, model_vgg, model_cc, epoch, args) # calc test accuracy
        accy_calc_time = datetime.now() - accy_time_start
        print("accy calculation took.... ", str(accy_calc_time))
        print("test_accy: ", test_waccy)
        print("test_loss: ", test_loss)
        accy['train_loss'], accy['epoch'], accy['train_word_accy'] = epoch_loss, epoch, epoch_word_accy
        accy['test_loss'], accy['test_word_accy'] = test_loss, test_waccy
        accy['train_time'], accy['accy_calc_time'] = epoch_time.seconds/60, accy_calc_time.seconds/60
        test_scores.append(accy) 

        # save scores to file
        json.dump(test_scores, open(os.path.join(args.model_save_path, 'model_accuracy.json'), 'w'))

    # Save Checkpoint
    checkpoint = {'epoch': epoch,
                'state_dict_cap': model_cc.state_dict(),
                'optimizer_cap': optimizer.state_dict(),
                'scheduler_cap': scheduler.state_dict(),

                'state_dict_img': model_vgg.state_dict(),
                'optimizer_img': optimizer_vgg.state_dict(),
                'scheduler_img': scheduler_vgg.state_dict()
                }

    # Write Checkpoint to disk
    torch.save(checkpoint, os.path.join(args.model_save_path, 'checkpoint.pt'))

    # Save highest-scoring model
    if accy['CIDEr'] >= max([value['CIDEr'] for value in test_scores]):
        torch.save(checkpoint, os.path.join(args.model_save_path, 'best_model.pt'))


print('End Training at Epoch ', args.num_epoch -1)