# implementation of a unimodal vae
# does not define which encoder/decoder-pair is used i.e. can instantiate this class with spatial or spatio-temporal encoder/decoder-pair

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
import torch

class UnimodalVAE(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temporal_mode = config["temporal_mode"]
        if self.temporal_mode == "lstm":
            self.bidirectional = config["bidirectional_lstm"]
            self.num_timesteps = config["temporal_input_factor"]  # timesteps in block
            self.mean_lstm = nn.LSTM(input_size=config["mean_vector_dim"], hidden_size=config["mean_vector_dim"], num_layers=config["lstm_layers"], bidirectional=config["bidirectional_lstm"], batch_first=True)
            self.std_lstm = nn.LSTM(input_size=config["mean_vector_dim"], hidden_size=config["mean_vector_dim"], num_layers=config["lstm_layers"], bidirectional=config["bidirectional_lstm"], batch_first=True)

    def forward(self, x):
        if self.temporal_mode == "lstm":
            # returns FINAL hidden state of mean_lstm and std_lstm 
            frame_recons = []
            mean_lstm_states, std_lstm_states = self.encode(x, all_states=True)  # need all states since we use the hidden state at timestep i to reconstruct frame i
            for temporal_index in range(self.num_timesteps): 
                sample = self.reparameterize(mean_lstm_states[:, temporal_index], std_lstm_states[:, temporal_index])  
                reconstruction = self.decoder(sample)
                frame_recons.append(reconstruction)
            final_mean_lstm_state = mean_lstm_states[:, -1]
            final_std_lstm_state = std_lstm_states[:, -1]
            recons_tensor = torch.stack(frame_recons, dim=1)  # surrounding code does not expect a list but a tensor
            return recons_tensor, final_mean_lstm_state, final_std_lstm_state
        else:
            mean, std_dev = self.encoder(x)
            # here can't just use mean but have to sample from the latent space
            # (otherwise the training process is not exposed to the stoachsticity of the latent space i.e. the model degenerates to an AE)
            sample = self.reparameterize(mean, std_dev)  
            reconstruction = self.decoder(sample)
            return reconstruction, mean, std_dev
    
    def reparameterize(self, mean, std_dev):
        epsilon = torch.randn_like(std_dev)  # sample from a standard normal distribution
        return mean + epsilon * std_dev

    # in lstm temporal_mode use parameter all_states to control the following behaviour: 
    # - returns FINAL hidden states of mean_lstm and std_lstm by default 
    # - returns ALL hidden statesm of mean_lstm and std_lstm if all_states = True 
    def encode(self, x, all_states=False):
        if self.temporal_mode == "lstm":
            means_per_timestep = []  # for each timestep in block collect batch of means
            stds_per_timestep = []  # for each timestep in block collect batch of stds
            for temporal_index in range(self.num_timesteps): 
                batched_frames_i = x[:, temporal_index, : , :]  # for all items in batch grab the frame at timestep temporal_index
                batched_means, batched_stds = self.encoder(batched_frames_i)
                means_per_timestep.append(batched_means)
                stds_per_timestep.append(batched_stds)

            # convert lists to torch tensors
            means_seq = torch.stack(means_per_timestep, dim=1)  
            stds_seq = torch.stack(stds_per_timestep, dim=1) 

            # apply lstms onto the sequences of means and stds we just collected
            mean_lstm_states, _ = self.mean_lstm(means_seq)  # hidden states per timestep
            std_lstm_states, _ = self.std_lstm(stds_seq)  # hidden states per timestep

            # sum forward and backward hidden states in case bidirectional was used (else downstream computations fails due to shape issues)
            if self.bidirectional:
                # use view and not reshape because desired data layout doesn't require shifting data in memory (and in that case view is more performant)
                mean_lstm_states = mean_lstm_states.view(mean_lstm_states.shape[0], mean_lstm_states.shape[1], 2, mean_lstm_states.shape[2] // 2)
                mean_lstm_states = torch.mean(mean_lstm_states, dim=2)
                std_lstm_states = std_lstm_states.view(std_lstm_states.shape[0], std_lstm_states.shape[1], 2, std_lstm_states.shape[2] // 2)
                std_lstm_states = torch.mean(std_lstm_states, dim=2)

            if all_states:
                return mean_lstm_states, std_lstm_states  
            else:  # return states of final timesteps only
                return mean_lstm_states[:, -1], std_lstm_states[:, -1]   
        else:
            mean, std_dev = self.encoder(x)
            return mean, std_dev

    # evidence lower bound (as presented on page 5 in https://arxiv.org/pdf/1312.6114)
    # all returned quantities are reduced over batch_dimension
    def loss(self, x, beta, recon, mean, std_dev): 
        # reconstruction term
        recon_loss = nn.functional.mse_loss(x, recon)  # reduces over batch_dim using 'mean'
        # kl divergence term
        var = std_dev ** 2

        # CLAMP
        # the std_dev produced by encoder can be zero OR very very small (e.g. 1e-28) ... 
        # which as far as encoder is concerned is valid (this might be intentionally returned by the encoder) but here in vae loss this is not fine anymore
        # as it will lead to var containing zeros which will make log_var contain -inf which will make loss nan which will corrupt model weights. 
        # hence we need to clamp. but what should we clamp: std_dev or var? turns out, clamping std_dev is not good enough because squaring eps 
        # (when computing var) will be small to the point that it will result in -inf when applying log. hence we have to clamp var (and not std_dev).
        var = self.clamp(var)

        kl_loss = self.kl_to_isotropic_gaussian(mean, var)  # kl_loss for each item in batch
        batch_averaged_kl_loss = torch.mean(kl_loss)  # nn.functional.mse_loss reduces over batch_dim using mean (so do the same here)
        weighted_kl_loss = beta * batch_averaged_kl_loss  # scalar * scalar
        total_loss = weighted_kl_loss + recon_loss # scalar + scalars

        return total_loss, recon_loss, batch_averaged_kl_loss
    
    # reutrns kl divergence of gaussian distribution defined by mean and var to an isotropic gaussian.
    # is batched i.e. takes in a batch of means and a batch of vars and returns a batch of kl divergences.
    def kl_to_isotropic_gaussian(self, mean, var):
        return - 0.5 * torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1)
    
    # clamp up all elems of a vector that are 0 to a small number
    def clamp(self, vector):
        eps = 0.00001
        vector[vector==0.0] = eps
        return vector