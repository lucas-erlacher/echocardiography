# a multimodal extractor encodes/reconstructs multimodal data. 

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from models.components.extractor.unimodal_extractors.unimodal_vae import UnimodalVAE
from models.components.extractor.unimodal_extractors.unimodal_mmvm_vae import UnimodalMMVMVAE
import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from helpers import *
from models.components.component_helpers import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class AbstractMultimodalExtractor(ABC, nn.Module):
    # encoding + decoding
    @abstractmethod
    def forward(self, x):
        pass

    # std_dev is always returned
    # if sample=True we return a sample from the learnt distribution else we return the mean
    @abstractmethod
    def encode(self, x, sample=False):
        pass

    # cross_val switches between cross_validation and full_data_training mode
    def train_component(self, full_model, cross_val, loader, train_data_logger, train_data_evaluator, validation_data_evaluator, beta_scheduler):
        model = full_model.extractor.to(self.config["device"])  # train all unimodal extractors in lockstep
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr_extractor"])
        for epoch_num in range(self.config["num_epochs_extractor"]):
            beta = beta_scheduler.get_beta(epoch_num)  # current approach is to update the beta once per epoch
            for batch_idx, batch in enumerate(tqdm(loader)):
                clips, _ = batch
                clips = [tensor.to(self.config["device"]) for tensor in clips]
                recon, random_frames, mean, std_dev = model(clips)
                # compute loss for each unimodal extractor
                loss_info = self.compute_loss(random_frames, beta, recon, mean, std_dev)
                # unpack total losses
                total_losses = [info[0] for info in loss_info]
                # backward pass 
                self.backpropagate(total_losses)
                # gradient logging (has to happen before we call zero_grad)
                if cross_val and (batch_idx == 0) and (epoch_num % self.config["evaluation_freq_extractor"] == 0):
                    models = []
                    for vae in model.unimodal_extractors.values():
                        l = [vae.encoder, vae.decoder]  # one enc - dec pair per view
                        models.append(l)
                    train_data_logger.log_gradients(models)  # it's fitting to log grad hists to train_data tb as these plots are generated using train data
                    del models
                optimizer.step()  # optimizer covers all unimodal extractors so one step updates all unimodal extractors
                optimizer.zero_grad()
                if cross_val and (batch_idx == 0) and (epoch_num % self.config["evaluation_freq_extractor"] == 0):
                    # it is interesting to see eval metrics (loss components, latent space visualizations, etc.) on both train and validation data
                    train_data_evaluator.extractor_eval(model, beta)
                    validation_data_evaluator.extractor_eval(model, beta)
                del clips, recon, random_frames, mean, std_dev, total_losses
            if check_exit(): return 
        return 

    # abstracts away the complexity of calling the right loss function (with the right parameters) depending on which vae type has been instantiated. 
    # returns a list of loss information tuples (which contain the different loss components)
    def compute_loss(self, random_frames, beta, recon, mean, std_dev):
        loss_info = None
        if self.config["vae_type"] == "regular":  # here we return a separate loss for each modality
            loss_info = [UnimodalVAE.loss(self.unimodal_extractors[view_key], random_frames[view_key], beta, recon[view_key], mean[view_key], std_dev[view_key]) for view_key in self.config["view_names"]]
        elif self.config["vae_type"] == "mmvm":  # here we return only return one loss for the entire extractor
            # the self reference in the following method call is only used for non-modality specific ops i.e. using any of the extractors (e.g. the first) is fine
            first_view_name = self.config["view_names"][0]
            first_extractor = self.unimodal_extractors[first_view_name]  
            loss_info = UnimodalMMVMVAE.loss(first_extractor, random_frames, beta, recon, mean, std_dev, self.config["view_names"])
            # P_1: since the callers of compute loss expect a list of 3 tuples (UnimodalMMVMVAE.loss returns a scalar, a list and a scalar
            # for to reasons outlined on top of the method implementation) we need to convert the return data into this form here
            loss_info = [(loss_info[0], recon_loss, loss_info[2]) for recon_loss in loss_info[1]]
        return loss_info
        
    # abstracts away the complexity of backpropagating the right way depending on which vae type has been instantiated
    def backpropagate(self, total_loss):
        if self.config["vae_type"] == "regular":
            for loss in total_loss: loss.backward()  
        elif self.config["vae_type"] == "mmvm":  
            # here we only need to backprop once (on any loss tensor) since here all total losses are the same (which can be seen at P_1 in the current file)
            total_loss[0].backward()

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir + '/extractor.pth')

    # does not return a loaded model but loads the data into self
    def load(self, path):
        return load_weights(path + '/extractor.pth', self, self.config["device"])  # load weights at path into current instance

    def get_summary(self):
        pass

    # TODO: maybe "confidence_score" could be another method that all multimodal encoders have to offer