import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class HypercolumnVgg(nn.Module):
    """
    Hypercolumn representation of image at pixel level using VGG16
    convolutional blocks as feature extractor. Feature maps at different
    convolution blocks are upsampled using bilinear upsampling to ensure the
    spatial resolution is the same to the input image. The output is a tensor
    with shape NCHW.
    """
    def __init__(self, n_conv_blocks=3):
        """
        :param n_conv_blocks: number of VGG16 convolutional blocks used for
        feature extraction.
        """
        super(HypercolumnVgg, self).__init__()
        self.features = nn.ModuleList(vgg16(weights=VGG16_Weights.DEFAULT).features)
        self.n_conv_blocks = n_conv_blocks
        self.feature_maps_index = [3, 8, 15, 22, 29]

    def forward(self, x):
        feature_maps_index = self.feature_maps_index[:self.n_conv_blocks]
        size = (x.size(2), x.size(3))
        feature_maps = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in feature_maps_index:
                feature_maps.append(x)
            if i == feature_maps_index[-1]:
                break
        features = []
        for map in feature_maps:
            upsample = F.interpolate(map, size=size, mode='bilinear', align_corners=True)
            features.append(upsample)
        outputs = torch.cat(features, 1)
        return outputs
