import os

import torch
import torch.nn as nn
from . import lpips as backbone

import pdb


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, net="vgg16"):
        super(LPIPS, self).__init__()

        version = "0.1"

        self.pnet_type = net
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net = "vgg"
            net_type = backbone.vgg16
            self.channels = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = backbone.alexnet
            self.channels = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = backbone.squeezenet
            self.channels = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.channels)

        self.net = net_type()

        self.lin0 = NetLinLayer(self.channels[0])
        self.lin1 = NetLinLayer(self.channels[1])
        self.lin2 = NetLinLayer(self.channels[2])
        self.lin3 = NetLinLayer(self.channels[3])
        self.lin4 = NetLinLayer(self.channels[4])
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.channels[5])
            self.lin6 = NetLinLayer(self.channels[6])
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)

        model_path = os.path.dirname(__file__) + f"/models/v{version}/{net}.pth"
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

    def forward(self, input):
        image_0 = input[0:1]
        image_1 = input[1:2]

        # Move input values from [0,1] [-1, +1]
        image_0 = 2 * image_0 - 1
        image_1 = 2 * image_1 - 1

        input_0 = self.scaling_layer(image_0)
        input_1 = self.scaling_layer(image_1)

        with torch.no_grad():
            output_0 = self.net.forward(input_0)
            output_1 = self.net.forward(input_1)
        features_0, features_1, difference = {}, {}, {}

        for k in range(self.L):
            features_0[k] = normalize_tensor(output_0[k])
            features_1[k] = normalize_tensor(output_1[k])

            difference[k] = (features_0[k] - features_1[k]) ** 2

        # features_0[0].size() -- [1, 64, 64, 64]  -- 256K
        # features_0[1].size() -- [1, 128, 32, 32] -- 128K
        # features_0[2].size() -- [1, 256, 16, 16] -- 64k
        # features_0[3].size() -- [1, 512, 8, 8] -- 32k
        # features_0[4].size() -- [1, 512, 4, 4] -- 8K

        res = [upsample(self.lins[k](difference[k]), out_HW=image_0.shape[2:]) for k in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1):
        super(NetLinLayer, self).__init__()

        layers = [
            nn.Dropout(),
        ]
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_model(model_name="squeeze"):
    model = LPIPS(net=model_name)
    return model
