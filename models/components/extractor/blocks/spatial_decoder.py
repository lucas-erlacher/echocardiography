# 2D convolution based decoder
# it's structure mirrors the structure of spatial_encoder.py

import torch.nn as nn

class SpatialDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.act_func = nn.ReLU()

        temporal_input_factor = self.config["temporal_input_factor"] if (self.config["temporal_mode"] == "channel") else 1
        temporal_parameter_factor = self.config["temporal_parameter_factor"] if (self.config["temporal_mode"] == "channel") else 1

        self.upconv_1 = nn.ConvTranspose2d(in_channels=self.config["final_channel_depth"], out_channels=16 * temporal_parameter_factor, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.optional_convs_1 = nn.ModuleList([nn.Conv2d(in_channels=16 * temporal_parameter_factor, out_channels=16 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_1 = nn.BatchNorm2d(16 * temporal_parameter_factor)
        self.upconv_2 = nn.ConvTranspose2d(in_channels=16 * temporal_parameter_factor, out_channels=8 * temporal_parameter_factor, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.optional_convs_2 = nn.ModuleList([nn.Conv2d(in_channels=8 * temporal_parameter_factor, out_channels=8 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_2 = nn.BatchNorm2d(8 * temporal_parameter_factor)
        self.upconv_3 = nn.ConvTranspose2d(in_channels=8 * temporal_parameter_factor, out_channels=4 * temporal_parameter_factor, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.optional_convs_3 = nn.ModuleList([nn.Conv2d(in_channels=4 * temporal_parameter_factor, out_channels=4 * temporal_parameter_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_3 = nn.BatchNorm2d(4 * temporal_parameter_factor)
        self.upconv_4 = nn.ConvTranspose2d(in_channels=4 * temporal_parameter_factor, out_channels=1 * temporal_input_factor, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.optional_convs_4 = nn.ModuleList([nn.Conv2d(in_channels=1 * temporal_input_factor, out_channels=1 * temporal_input_factor, kernel_size=3, padding=1) for _ in range(self.config["parameter_factor"])])
        self.bn_4 = nn.BatchNorm2d(1 * temporal_input_factor)
        self.upconvs = nn.ModuleList([self.upconv_1, self.upconv_2, self.upconv_3, self.upconv_4])
        self.optional_convs = nn.ModuleList([self.optional_convs_1, self.optional_convs_2, self.optional_convs_3, self.optional_convs_4])
        self.bns = nn.ModuleList([self.bn_1, self.bn_2, self.bn_3, self.bn_4])

        self.expansion_factor = 2 ** len(self.upconvs)  # factor by how much upconvs will expand input (assuming each upconv will double dimensions)
        self.fc_1 = nn.Linear(self.config["mean_vector_dim"], self.config["final_channel_depth"] * int(self.config["frame_sidelen"] / self.expansion_factor) ** 2)

    def forward(self, x):
        x = self.fc_1(x)  # blow up sampled hidden vector s.t. it can then be reshaped into tensor that 
                          # upconvs can then turn into a decoder-output that has correct dimentsions. 
        x = x.reshape(self.config["batch_size_extractor"], self.config["final_channel_depth"], int(self.config["frame_sidelen"] / self.expansion_factor), int(self.config["frame_sidelen"] / self.expansion_factor))
        for idx, upconv in enumerate(self.upconvs):
            # definitely do the upconv
            x = upconv(x)
            # optionally do more convs (depending on the parameter_factor)
            optional_convs = self.optional_convs[idx]
            for opt_conv in optional_convs:
                x = opt_conv(x)
            x = self.bns[idx](x)
            x = self.act_func(x)
        # regensurg dataset wrapper returns frames without a channel dimension (since it is 1 implicitly) so rm channel dim before recon goes into loss
        x = x.squeeze(1)
        return x