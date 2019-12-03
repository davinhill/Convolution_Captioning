import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# ======================================================
    # VGG16 Model for extracting image features
# ======================================================
vgg_pretrained = models.vgg16(pretrained = True)
class vgg_extraction(nn.Module):
    def __init__(self, img_feat):
        super(vgg_extraction, self).__init__()

        self.feature_layers = vgg_pretrained.features  # all convolution layers in VGG, final layer is maxpool
        
        layers = []
        layers.append(vgg_pretrained.classifier[:-1])  # all fc layers, excluding final linear layer
        layers.append(nn.Linear(4096, img_feat)) # apply fc layer. final output is 512 dim.
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        output_conv = self.feature_layers(x)  # output of convolutions layers (for attention calculations)
        output_fc = self.fc(torch.flatten(output_conv, 1))  # output of fc layers (for image embedding)
        return output_conv, output_fc


# ======================================================
    # Attention Model
# ======================================================
class AttnBlock(nn.Module):
    def __init__(self, vocab_size):
        super(AttnBlock, self).__init__()


    def forward(self, x):

        return x
        
        
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
    # Convolution Captioning Model
# ======================================================
class conv_captioning(nn.Module):
    def __init__(self, vocab_size, kernel_size, num_layers, dropout_p, word_feat, input_feat):
        super(conv_captioning, self).__init__()

        # Embedding Layers
        self.word_embedding0 = nn.Embedding(vocab_size, word_feat)
        self.word_embedding1 = nn.utils.weight_norm(nn.Linear(word_feat, word_feat))

        # Convolution layers
        conv_layers = []
        for i in range(num_layers-1):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
            if i == 0:
                conv_layers.append(ConvBlock(input_feat, input_feat, kernel_size, dropout_p))
            else:
                conv_layers.append(ConvBlock(int(input_feat/2), input_feat, kernel_size, dropout_p))
        self.conv_n = nn.Sequential(*conv_layers)

        # Classification layers
        self.fc1 = nn.utils.weight_norm(nn.Linear(int(input_feat / 2), int(input_feat / 4)))
        self.fc2 = nn.utils.weight_norm(nn.Linear(int(input_feat / 4), vocab_size))

    def forward(self, caption_tknID, img_fc):
        
        # Embedding Layers
        word_embed = self.word_embedding0(caption_tknID)
        word_embed = self.word_embedding1(word_embed)
        # word_embed: n x (max_cap_len) x 512
        # image_embed: n x 512

        # Reshape image embedding & concatenate with word embedding
        img_embed = img_fc.unsqueeze(1).expand(-1, word_embed.shape[1], -1)
        input_embed = torch.cat((word_embed, img_embed), 2).transpose(1, 2) # n x 1024 x (max_cap_len)

        # convolution layers
        x = self.conv_n(input_embed) # n x 512 x max_cap_len
        #x = F.dropout(x, p = self.dropout_p)

        # note: I'm following the source code, however I'm not sure why there isn't a relu between the two FC layers? Also, why the extra 2 dropout layers?
        # classifier layers
        x = x.transpose(1,2)  # n x max_cap_len x 512
        x = self.fc1(x) # n x max_cap_len x 256
        #x = F.dropout(x, p = self.dropout_p)
        x = F.relu(x)
        
        x = self.fc2(x) # n x max_cap_len x vocab_size
        x = x.transpose(1,2) # n x vocab_size x max_cap_len

        return x # n x vocab_size x max_cap_len
        
        
        
        
        
        
        
        
