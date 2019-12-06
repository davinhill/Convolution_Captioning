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


        
        
class ConvBlock(nn.Module):
    def __init__(self, input_feat, output_feat, kernel_size, dropout_p, is_bidirectional):
        super(ConvBlock, self).__init__()       
        if is_bidirectional:
            self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat/2, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        else:
            self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        self.dropout = nn.Dropout(p = dropout_p)
        self.kernel_size = kernel_size
        self.downsample = nn.Linear(input_feat, int(output_feat/2))
        self.is_bi = is_bidirectional

    def forward(self, x):
        identity = x   # skip connection
        x = self.conv(x)
        
        #Bidirectional implementation
        if self.is_bi:
            x1 = identity
            x1 = torch.flip(x1, [2])
            x1 = self.conv(x1)
            x1 = torch.flip(x1, [2])
            x = x + x1
            
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
    def __init__(self, input_feat, output_feat, kernel_size, dropout_p, word_feat, is_bidirectional):
        super(AttnBlock, self).__init__()
        if is_bidirectional:
            self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat/2, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        else:
            self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = kernel_size - 1, stride = 1))
        self.dropout = nn.Dropout(p = dropout_p)
        self.kernel_size = kernel_size
        self.downsample = nn.Linear(input_feat, int(output_feat/2))

        # Attention Layers
        attn_channels = 512 # number of channels in vgg convolution layer output, for use in attention
        self.attn_fc1 = nn.Linear(word_feat, attn_channels)
        self.attn_fc2 = nn.Linear(attn_channels, word_feat)
        self.is_bi = is_bidirectional

    def forward(self, list_input):
        
        x, word_embed, img_conv, prev_attn = list_input[0], list_input[1], list_input[2], list_input[3]

        identity = x   # skip connection
        x = self.conv(x)
        
        #Bidirectional implementation
        if self.is_bi:
            x1 = identity
            x1 = torch.flip(x1, [2])
            x1 = self.conv(x1)
            x1 = torch.flip(x1, [2])
            x = x + x1
        
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
    def __init__(self, vocab_size, kernel_size, num_layers, dropout_p, word_feat, input_feat, attn, is_bidirectional = False):
        super(conv_captioning, self).__init__()
        
        self.attn = attn
        # Embedding Layers
        self.word_embedding0 = nn.Embedding(vocab_size, word_feat)
        self.word_embedding1 = nn.utils.weight_norm(nn.Linear(word_feat, word_feat))

        # Convolution / Attention layers
        conv_layers = []
        if self.attn:
            for i in range(num_layers):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
                if i == 0:
                    conv_layers.append(AttnBlock(input_feat, input_feat, kernel_size, dropout_p, word_feat, is_bidirectional))
                else:
                    conv_layers.append(AttnBlock(int(input_feat/2), input_feat, kernel_size, dropout_p, word_feat, is_bidirectional))

        else:
            for i in range(num_layers-1):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
                if i == 0:
                    conv_layers.append(ConvBlock(input_feat, input_feat, kernel_size, dropout_p, is_bidirectional))
                else:
                    conv_layers.append(ConvBlock(int(input_feat/2), input_feat, kernel_size, dropout_p, is_bidirectional))
        self.conv_n = nn.Sequential(*conv_layers)

        # Classification layers
        self.fc1 = nn.utils.weight_norm(nn.Linear(int(input_feat / 2), int(input_feat / 4)))
        self.fc2 = nn.utils.weight_norm(nn.Linear(int(input_feat / 4), vocab_size))
        self.drop1 = nn.Dropout(p = dropout_p)
        

    def forward(self, caption_tknID, img_fc, img_conv):
        
        # Embedding Layers
        word_embed = self.word_embedding0(caption_tknID)
        word_embed = self.word_embedding1(word_embed)
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
        
        
        
        
        
        
        
        
