# 2D convolution based encoder

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
import torch

class SpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act_func = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # always pool the same way

        temporal_input_factor = self.config["temporal_input_factor"] if (self.config["temporal_mode"] == "channel") else 1
        temporal_parameter_factor = self.config["temporal_parameter_factor"] if (self.config["temporal_mode"] == "channel") else 1

        self.conv_1 = nn.Conv2d(in_channels=temporal_input_factor, out_channels=2 * temporal_parameter_factor, kernel_size=3, padding=1)
        self.optional_convs_1 = nn.ModuleList([nn.Conv2d(in_channels=2 * temporal_parameter_factor, out_channels=2 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_1 = nn.BatchNorm2d(2 * temporal_parameter_factor)
        self.conv_2 = nn.Conv2d(in_channels=2 * temporal_parameter_factor, out_channels=4 * temporal_parameter_factor, kernel_size=3, padding=1)
        self.optional_convs_2 = nn.ModuleList([nn.Conv2d(in_channels=4 * temporal_parameter_factor, out_channels=4 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_2 = nn.BatchNorm2d(4 * temporal_parameter_factor)
        self.conv_3 = nn.Conv2d(in_channels=4 * temporal_parameter_factor, out_channels=8 * temporal_parameter_factor, kernel_size=3, padding=1)
        self.optional_convs_3 = nn.ModuleList([nn.Conv2d(in_channels=8 * temporal_parameter_factor, out_channels=8 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_3 = nn.BatchNorm2d(8 * temporal_parameter_factor)
        self.conv_4 = nn.Conv2d(in_channels=8 * temporal_parameter_factor, out_channels=self.config["final_channel_depth"], kernel_size=3, padding=1)
        self.optional_convs_4 = nn.ModuleList(
            [nn.Conv2d(in_channels=self.config["final_channel_depth"], out_channels=self.config["final_channel_depth"], kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_4 = nn.BatchNorm2d(self.config["final_channel_depth"])

        self.convs = nn.ModuleList([self.conv_1, self.conv_2, self.conv_3, self.conv_4])
        self.optional_convs = nn.ModuleList([self.optional_convs_1, self.optional_convs_2, self.optional_convs_3, self.optional_convs_4])
        self.bns = nn.ModuleList([self.bn_1, self.bn_2, self.bn_3, self.bn_4])

        reduction_factor = 2 ** len(self.convs)  # assumes pool (with 2x reduction) after each conv

        # warning printout if mean_vector_dim is larger than reshaped_activation_size
        reshaped_activation_size = self.config["final_channel_depth"] * int(self.config["frame_sidelen"] / reduction_factor) ** 2

        self.mean_generator = nn.Linear(reshaped_activation_size, self.config["mean_vector_dim"])
        self.stddev_generator = nn.Linear(reshaped_activation_size, self.config["mean_vector_dim"])

    # takes in an image (as this is a spatial encoder). 
    def forward(self, x):
        # regensurg dataset wrapper returns frames without a channel dimension (since it is 1 implicitly) so add it
        if not self.config["temporal_mode"] == "channel": 
            x = x.unsqueeze(1) 
        for idx, conv in enumerate(self.convs): 
            # definitely do one conv
            x = conv(x)
            # optionally do more convs (depending on the parameter_factor)
            optional_convs = self.optional_convs[idx]
            for opt_conv in optional_convs:
                x = opt_conv(x)
            x = self.bns[idx](x)  
            x = self.act_func(x)
            x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # flatten all dims excluding batch size
        mean = self.mean_generator(x)
        # sigmoid to ensure positivity (relying on squaring of stddev in elbo is not enough as other methods (e.g. encode) use stddev i.e. expect positivity)
        stddev = torch.exp(self.stddev_generator(x))
        stddev = stddev + 0  # hack to avoid torch complaining about the exp() being an in-place operation
        return mean, stddev