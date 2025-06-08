# OUTDATED: old code that has not been tested with recent changes to other parts of the code i.e. if time fix it (or delete it)

# encoder based on pretrained models (e.g. ResNet) i.e. take pretrained weights and fine tune them on our data

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
import torch
import torchvision.models as models
from helpers import *

class PretrainedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # load resnet weights
        self.pretrained_model = models.resnet18(weights=None)
        state_dict = torch.load("/cluster/home/elucas/thesis/pretrained/resnet18-f37072fd.pth", map_location='cpu')
        self.pretrained_model.load_state_dict(state_dict)

        # freeze all pretrained weights
        if self.config["freeze_pretrained"]:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # replace final pretrained layer (first save num_feats for later)
        num_feats = self.pretrained_model.fc.in_features  # 512 in case of resnet-18
        self.pretrained_model.fc = nn.Linear(num_feats, num_feats)

        self.mean_generator = nn.Linear(num_feats, self.config["mean_vector_dim"])
        self.stddev_generator = nn.Linear(num_feats, self.config["mean_vector_dim"])

    # takes in an image (as this is a spatial encoder). 
    def forward(self, x):
        # regensurg dataset wrapper returns frames without a channel dimension (since it is 1 implicitly) so add it
        x = x.unsqueeze(1) 
        x = x.repeat(1, 3, 1, 1)  # resnet expects 3 channels i.e. copy greyscale image 3 times along channel dim 
        x = self.pretrained_model(x)
        # generate mean and std_dev
        mean = self.mean_generator(x)
        # sigmoid to ensure positivity (relying on squaring of stddev in elbo is not enough as other methods (e.g. encode) use stddev i.e. expect positivity)
        stddev = torch.exp(self.stddev_generator(x))
        stddev = stddev + 0  # hack to avoid torch complaining about the exp() being an in-place operation
        return mean, stddev  