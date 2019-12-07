import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# ======================================================
# VGG16 Model for extracting image features
# ======================================================
class vgg_extraction(nn.Module):
    def __init__(self, img_feat):
        super(vgg_extraction, self).__init__()

        vgg_pretrained = models.vgg16(pretrained=True)

        # all convolution layers in VGG, final layer is maxpool
        self.feature_layers = vgg_pretrained.features

        layers = []
        # all fc layers, excluding final linear layer
        layers.append(vgg_pretrained.classifier[:-1])
        # apply fc layer. final output is 512 dim.
        layers.append(nn.Linear(4096, img_feat))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # output of convolutions layers (for attention calculations)
        output_conv = self.feature_layers(x)
        # output of fc layers (for image embedding)
        output_fc = self.fc(torch.flatten(output_conv, 1))

        return output_conv, output_fc




# ======================================================
# Pretrained resnet18 model for image extraction
# ======================================================
class resnet_extraction(nn.Module):
    def __init__(self, img_feat):
        super(resnet_extraction, self).__init__()

        resnet_pretrained = models.resnet18(pretrained=True)

        # convolution output, for use with attention
        # 7 x 7 x 512
        self.feature_layers = nn.Sequential(*(list(resnet_pretrained.children())[:-2]))
        
        self.pool = nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)
        self.fc = nn.Linear(512, img_feat)

    def forward(self, x):
        # output of convolutions layers (for attention calculations)
        output_conv = self.feature_layers(x)

        # output of fc layers (for image embedding)
        output_fc = self.pool(output_conv)
        output_fc = torch.flatten(output_fc, 1)
        output_fc = self.fc(output_fc)

        return output_conv, output_fc





# ======================================================
# Pretrained densenet for image extraction
# ======================================================
class densenet_extraction(nn.Module):
    def __init__(self, img_feat):
        super(densenet_extraction, self).__init__()

        densenet_pretrained = models.densenet121(pretrained=True)

        # 7 x 7 x 1024
        self.feature_layers = densenet_pretrained.features
        self.downsample = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1, stride = 1, padding = 0)
        self.fc = nn.Linear(1024, img_feat)

    def forward(self, x):
        # output of convolutions layers (for attention calculations)
        x = self.feature_layers(x)
        output_conv = self.downsample(x)

        # output of fc layers (for image embedding)
        output_fc = F.adaptive_avg_pool2d(x, (1, 1))
        output_fc = torch.flatten(output_fc, 1)
        output_fc = self.fc(output_fc)

        return output_conv, output_fc
