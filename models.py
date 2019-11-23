import torch
import torch.nn as nn
from torchvision import models



# ======================================================
    # VGG16 Model for extracting image features
# ======================================================
vgg_pretrained = models.vgg16(pretrained = True, progress = False)
class vgg_extraction(nn.Module):
    def __init__(self):
        super(vgg_extraction, self).__init__()

        self.feature_layers = vgg_pretrained.features  # all convolution layers in VGG, final layer is maxpool
        
        layers = []
        layers.append(vgg_pretrained.classifier[:-1])  # all fc layers, excluding final linear layer
        layers.append(nn.Linear(4096, 512)) # apply fc layer. final output is 512 dim.
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        output_conv = self.feature_layers(x)  # output of convolutions layers (for attention calculations)
        output_fc = self.fc(output_conv)  # output of fc layers (for image embedding)
        return output_conv, output_fc


# ======================================================
    # Attention Model
# ======================================================
class attention(nn.Module):
    def __init__(self, vocab_size):
        super(attention, self).__init__()


    def forward(self, x):

        return x
        
        
        
        

# ======================================================
    # Convolution Captioning Model
# ======================================================
class conv_captioning(nn.Module):
    def __init__(self, vocab_size):
        super(conv_captioning, self).__init__()

        self.word_embedding0 = nn.embedding(vocab_size, 512)
        self.word_embedding1 = nn.embedding(512, 512)

    def forward(self, caption_tknID, img_fc):

        word_embed = self.word_embedding0(caption_tknID)
        word_embed = self.word_embedding1(word_embed)

        input_embed = torch.cat((word_embed, img_fc), 1)


        return caption
        
        
        
        
        
        
        
        
