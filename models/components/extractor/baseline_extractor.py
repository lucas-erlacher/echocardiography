import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
from models.components.extractor.abstract_multimodal_extractor import AbstractMultimodalExtractor
from models.components.extractor.unimodal_extractors.unimodal_vae import UnimodalVAE
from models.components.extractor.unimodal_extractors.unimodal_mmvm_vae import UnimodalMMVMVAE
from models.components.extractor.blocks.spatial_encoder import SpatialEncoder
from models.components.extractor.blocks.pretrained_encoder import PretrainedEncoder
from models.components.extractor.blocks.spatial_decoder import SpatialDecoder
import random
import torch
from helpers import *
from tqdm import tqdm
from models.components.component_helpers import *

class BaselineExtractor(AbstractMultimodalExtractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vae_type = self.config["vae_type"]
        self.unimodal_extractors = nn.ModuleDict()  # need to use ModuleDict otherwise pytorch won't recognize the parameters of unimodal extractors
        for view_name in self.config["view_names"]:
            # construct the encoder and decoder
            if self.config["encoder"] == "spatial":
                enc = SpatialEncoder(config)
            elif self.config["encoder"] == "pretrained":
                enc = PretrainedEncoder(config)
            dec = SpatialDecoder(config)
            # wrap enc and dec into a unimodal vae which will be responsible for a certain echo-view
            if self.vae_type == "regular":
                self.unimodal_extractors[view_name] = UnimodalVAE(enc, dec, config) 
            elif self.vae_type == "mmvm":
                self.unimodal_extractors[view_name] = UnimodalMMVMVAE(enc, dec, config)

    # used for training of extractor module. 
    # also need to return the selected random frames because otherwise loss function of training loop can't know which frames were randomly selected. 
    def forward(self, x): 
        reconstructions = {}
        batched_random_frames = {}
        means = {}
        std_devs = {}
        for view_idx, view_key in enumerate(self.unimodal_extractors):
            fitting_clips = x[view_idx] 
            if self.config["temporal_mode"] == None:
                random_frames = self.__get_random_frames(fitting_clips)
            else:
                random_frames = fitting_clips
            reconstruction, mean, std_dev = self.unimodal_extractors[view_key](random_frames)  # invoke forward method
            reconstructions[view_key] = reconstruction
            batched_random_frames[view_key] = random_frames 
            means[view_key]= mean
            std_devs[view_key] = std_dev
        return reconstructions, batched_random_frames, means, std_devs

    # takes in a set of videos (from different views). 
    def encode(self, x, sample=False):
        means = []
        std_devs = []
        samples = []
        for view_idx, view_key in enumerate(self.unimodal_extractors):
            fitting_clips = x[view_idx] 
            if self.config["temporal_mode"] == None:
                random_frames = self.__get_random_frames(fitting_clips)
            else:
                random_frames = fitting_clips
            mean, std_dev = self.unimodal_extractors[view_key].encode(random_frames)
            means.append(mean)
            std_devs.append(std_dev)
            if sample:
                samples.append(self.unimodal_extractors[view_key].reparameterize(mean, std_dev))
        if sample == False:
            return torch.cat(means, dim=1), torch.cat(std_devs, dim=1)  # return mean and std_dev
        else:
            return torch.cat(samples, dim=1), torch.cat(std_devs, dim=1)  # return samples

    def get_summary(self):
        s = ""
        s += "encoder: " + str(self.config["encoder"]) + "\n"
        s += "final channel depth: " + str(self.config["final_channel_depth"]) + "\n"
        s += "freeze_pretrained: " + str(self.config["freeze_pretrained"]) + "\n"
        s += "mean vector dim: " + str(self.config["mean_vector_dim"]) + "\n"
        s += "parameter_factor: " + str(self.config["parameter_factor"]) + "\n"  
        s += "vae type: " + str(self.config["vae_type"]) + "\n"
        if self.config["vae_type"] == "mmvm":
            s += "alpha: " + str(self.config["alpha"]) 
        if self.config["temporal_mode"] == "lstm":
            s += "lstm layers: " + str(self.config["lstm_layers"]) + "\n"
            s += "bidirectional_lstm: " + str(self.config["bidirectional_lstm"]) + "\n" 
        return s

    # given a batch of clips randomly select a random frame from each clip.
    # the different echo-views are NOT handled by this method i.e. they need to be iterated over outside of this method. 
    def __get_random_frames(self, x):
        # in case dataset is set up with frame_block_size=1 the resulting blocks will not contain a frame_dim (don't ask me why) in which case we need to manually
        # give them a frame_dim (downstream ops expect that) which can be done here since all blocks that are passed to extractor eventually go through this method. 
        #
        # the more elegant solution would be to fix this issue of the disappearing frame_dim in the dataset code but that code is not mine and 
        # at the moment I do not have the time to read and understand a relatively compilicated repo just to make this fix slightly more elegant. 
        if len(x.shape) == 3: x = x.unsqueeze(1)
        # sample a random frame from each clip
        random_frames = [clip[random.randint(0, clip.size(0) - 1)] for clip in x]
        random_frames_tensor = torch.stack(random_frames)
        return random_frames_tensor