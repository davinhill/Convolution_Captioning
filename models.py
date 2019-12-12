import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np


        
class ConvBlock(nn.Module):
    def __init__(self, input_feat, output_feat, kernel_size, dropout_p):
        super(ConvBlock, self).__init__()       

        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        self.dropout = nn.Dropout(p = dropout_p)
        self.kernel_size = kernel_size
        self.downsample = nn.Linear(input_feat, int(output_feat/2))

    def forward(self, x):
        identity = x   # skip connection
        x = self.conv(x) 
        x = self.dropout(x)
        x = F.glu(x, 1)  # reduces feature dimenion by 1/2. 
        x = x[:,:,:-(self.kernel_size-1)]  # truncate by kernel_size - 1

        # downsample the identity if there is a dimension mismatch (in the first conv layer)
        if identity.shape != x.shape:
            identity = identity.transpose(1, 2)
            identity = self.downsample(identity)
            identity = identity.transpose(1, 2)

        return x + identity  

# ======================================================
    # Attention Model
# ======================================================
class AttnBlock(nn.Module):
    def __init__(self, input_feat, output_feat, kernel_size, dropout_p, word_feat):
        super(AttnBlock, self).__init__()

        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        self.dropout = nn.Dropout(p = dropout_p)
        self.kernel_size = kernel_size
        self.downsample = nn.Linear(input_feat, int(output_feat/2))

        # Attention Layers
        attn_channels = 512 # number of channels in vgg convolution layer output, for use in attention
        self.attn_fc1 = nn.Linear(output_feat // 2 , attn_channels)
        self.attn_fc2 = nn.Linear(attn_channels, output_feat // 2)

    def forward(self, list_input):
        
        x, word_embed, img_conv, prev_attn = list_input[0], list_input[1], list_input[2], list_input[3]

        identity = x   # skip connection
        x = self.conv(x) 
        x = self.dropout(x)
        x = F.glu(x, 1)  # reduces feature dimenion by 1/2. 
        x = x[:,:,:-(self.kernel_size-1)]  # truncate by kernel_size - 1
        # n x 512 x max_cap_len

        # Attention ################## 
        identity_attn = x # n x 512 x max_cap_len

        # apply linear layer to x, then combine with word embedding
        x = self.attn_fc1(x.transpose(1, 2)) + word_embed # n x max_cap_len x 512
        # in the paper, this is denoted W(d_j)

        # flatten each attn channel
        attn_feat = img_conv.flatten(2) # n x 512 x 49
        # in the paper, this is denoted c_i

        # Calc softmax on each attn channel
        attn_score = torch.matmul(x, attn_feat) # n x max_cap_len x 49
        rs = attn_score.size()  # n x 512 x 49
        attn_score = attn_score.flatten(end_dim = 1) # (n * max_cap_len) x 49
        attn_score = F.softmax(attn_score, 1) # I don't necessarily think this is correct, at least according to the paper?
        attn_score = attn_score.reshape(rs) # n x 512 x 49
        x = torch.matmul(attn_score, attn_feat.transpose(1, 2)) # n x max_cap_len x 512

        # up/downsample x to revert it back to the convolution layer output dimensions
        x = self.attn_fc2(x)

        # Residual connection
        x = x.transpose(1, 2) + identity_attn # n x 512 x max_cap_len

        ##################

        # downsample the identity if there is a dimension mismatch (in the first conv layer)
        if identity.shape != x.shape:
            identity = identity.transpose(1, 2)
            identity = self.downsample(identity)
            identity = identity.transpose(1, 2)

        return [x + identity, word_embed, img_conv, attn_score]



# ======================================================
    # Convolution Captioning Model
# ======================================================
class conv_captioning(nn.Module):
    def __init__(self, vocab_size, kernel_size, num_layers, dropout_p, word_feat, input_feat, args):
        super(conv_captioning, self).__init__()

        self.attn = args.attention

        # Embedding Layers
        emb_layers = []
        if not args.use_glove: 
            # if using self-created word embedding
            word_embedding0 = nn.Embedding(vocab_size, word_feat)
            word_embedding1 = nn.utils.weight_norm(nn.Linear(word_feat, word_feat))

            if args.freeze_embed:
                print('Freezing embedding weights...')
                word_embedding0.weight.requires_grad = False  # freeze parameters

            emb_layers.append(word_embedding0)
            emb_layers.append(word_embedding1)
        else:
            # if using pretrained glove features:
            print('Loading glove features...')
            glove_feat = np.load('embed/glove_embeddings.npy')  # load glove features
            glove_feat = glove_feat[:vocab_size, :]  # truncate to vocab_size

            embed = nn.Embedding(vocab_size, 300)
            embed.weight.data.copy_(torch.from_numpy(glove_feat))  # load glove features into embedding layer

            if args.freeze_embed:
                print('Freezing embedding weights...')
                embed.weight.requires_grad = False  # freeze parameters

            emb_layers.append(embed)
            emb_layers.append(nn.utils.weight_norm(nn.Linear(300, word_feat)))
        self.embedding = nn.Sequential(*emb_layers)


        # Convolution / Attention layers
        conv_layers = []
        if self.attn:
            for i in range(num_layers):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
                if i == 0:
                    conv_layers.append(AttnBlock(input_feat, input_feat, kernel_size, dropout_p, word_feat))
                else:
                    conv_layers.append(AttnBlock(int(input_feat/2), input_feat, kernel_size, dropout_p, word_feat))

        else:
            for i in range(num_layers-1):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
                if i == 0:
                    conv_layers.append(ConvBlock(input_feat, input_feat, kernel_size, dropout_p))
                else:
                    conv_layers.append(ConvBlock(int(input_feat/2), input_feat, kernel_size, dropout_p))
        self.conv_n = nn.Sequential(*conv_layers)

        # Classification layers
        self.fc1 = nn.utils.weight_norm(nn.Linear(int(input_feat / 2), int(input_feat / 4)))
        self.fc2 = nn.utils.weight_norm(nn.Linear(int(input_feat / 4), vocab_size))
        self.drop1 = nn.Dropout(p = dropout_p)

    def forward(self, caption_tknID, img_fc, img_conv):

        # Embedding Layers
        word_embed = self.embedding(caption_tknID)
        # word_embed: n x (max_cap_len) x 512
        # image_embed: n x 512

        # Reshape image embedding & concatenate with word embedding
        img_embed = img_fc.unsqueeze(1).expand(-1, word_embed.shape[1], -1)
        input_embed = torch.cat((word_embed, img_embed), 2).transpose(1, 2) # n x 1024 x (max_cap_len)

        # convolution/attention layers
        attn_score = None
        if self.attn:
            list_output = self.conv_n([input_embed, word_embed, img_conv, attn_score]) 
            x, attn_score = list_output[0], list_output[3]
        else:
            x = self.conv_n(input_embed)  # n x 512 x max_cap_len

        # classifier layers
        x = x.transpose(1,2)  # n x max_cap_len x 512
        x = self.fc1(x) # n x max_cap_len x 256
        x = self.drop1(x)
        x = F.relu(x)
        
        x = self.fc2(x) # n x max_cap_len x vocab_size
        x = x.transpose(1,2) # n x vocab_size x max_cap_len

        return x, attn_score
        # x: n x vocab_size x max_cap_len
        # attn_scores: n x max_cap_len x 49
        
        
        
        
        
        
        
        
