# unimodal VAE that is based on the MMVM VAE paper (https://arxiv.org/pdf/2403.05300) i.e. this model is
# identical to unimodal_vae.py except for the loss function (which is proposed in the paper).

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
import torch
from models.components.extractor.unimodal_extractors.unimodal_vae import UnimodalVAE
import numpy as np

class UnimodalMMVMVAE(UnimodalVAE):
    def __init__(self, enc, dec, config):
        super().__init__(enc, dec, config)
        self.temporal_mode = config["temporal_mode"]
        self.alpha = config["alpha"]
    
    # RETURN VALUES:
    # - total loss (= reconstruction term and regularizing term): 
    #   loss of the entire extractor at once i.e. not only for a single unimodal_extractor. 
    #   reason for not computing individual extractor losses is the fact that the regularizer 
    #   is computed at the extractor level (see last point for why we are doing this). 
    # - reconstruction losses:
    #   unlike for the regularizer we can (and do) compute (and return) individual view level recon losses in order to at least 
    #   be able to track this loss on a view level (unlike regularizer and total_loss which can only be tracked on the extractor level)). 
    # - regularizer: 
    #   the computation of this quantity (R from page 5 of the paper) has been taken from the offical paper code. 
    #   in an effort to not negativley affect the performance of the proposed method we refrain from tweaking this code 
    #   (as seemingly small changes can have a large effect on learning performance). since the official code does not 
    #   compute the modality level components of the regularizer but instead only the entire regularizer for the whole extractor
    #   (and we have decided to not touch this code) we only return the extractor-level regularizer from this method. 
    #
    # IMPLEMENTATION COMMENTS: 
    # - the computation of the regularizer was taken from the paper code and the computation of the recon loss was implemented by me 
    #   in a way s.t. they both use the same reduction functions when reducing over the modalities (mean) and the batch (mean). 
    def loss(self, xs, beta, recons, means, std_devs, view_names): 
        vars = {k: v ** 2 for k, v in std_devs.items()}
        # clamp every variance of the batch (reason why we clamp var and not std_dev is same as in unimodal_vae.py)
        vars = {k: self.clamp(v) for k, v in vars.items()}

        ########  COMPUTE R (FROM PAGE 5 OF THE PAPER)  ########
        num_views = len(view_names)
        klds = []  # the code from the paper (which I do not want to mess with) is not collecting kls per modality but is collecting all of them in a flat list
        for view_key in view_names:  # for each encoder distrib collect a few KLDs (but append all kls that are computed for any encoder distrib into the same flat list "klds")
            dist_m = [means[view_key], vars[view_key]]  # this is the encoder distrib i.e. the distrib that we want to improve with gradients from this loss
            # add KLDs that pull encoder distrib towards the constituents of h(z | X) and thereby towards h(z | X)
            for view_key_tilde in view_names:
                dist_m_tilde = [means[view_key_tilde], vars[view_key_tilde]]
                kld_m_m_tilde = self.custom_kl(dist_m[0], dist_m[1], dist_m_tilde[0], dist_m_tilde[1])
                kld_m_m_tilde = kld_m_m_tilde.unsqueeze(1) 
                klds.append(kld_m_m_tilde * (1.0 - self.alpha))
            # add KLD that pulls encoder distrib towards N(0, 1)
            #
            # if alpha_weight is 0 (which it currently is) the following two lines are irrelevant
            # alpha_annealing can be used to in beginning of training have one KL per decoder that tries to "pull it towards" an isotropic gaussian
            # and then as training goes along this KL is less and less important until it is completely ignored after n steps (alpha_weight is then 0)
            kld_m = self.kl_to_isotropic_gaussian(dist_m[0], dist_m[1])
            klds.append(kld_m.unsqueeze(1) * self.alpha * num_views)
        # combine all the kls from the list "klds" in a way (implemented by the paper authors) which I think is equivalent to:
        # i) reduce the kls per modality: sum all the kls of one modality togehter 
        # ii) reduce over the modalities: average these sums togehter (i.e. the reduction over modalities is mean and not sum)
        modality_averaged_regularizer = torch.cat(klds, dim=1).sum(dim=1) / num_views  # combination of all kls of all modalities per batch
        # average over the batch (since nn.functional.mse_loss also averages over batch)
        batch_averaged_regularizer = modality_averaged_regularizer.mean(dim=0)  # fully reduced regularizer (i.e. a scalar)

        ########  RECON LOSS  ########
        recon_loss_sum = 0
        recon_losses = []
        # compute the recon loss for all views and sum them together 
        for view_key in view_names:
            loss = nn.functional.mse_loss(xs[view_key], recons[view_key])  # reduces over batch_dim using 'mean'
            recon_loss_sum += loss  
            recon_losses.append(loss)
        # in the end we want the combination of recon losses over modalities to be mean (and not sum) since that is how the paper code 
        # reduced over modalities in the regularizer (and we want to be consistent with that in order not to inflate the weight of recon loss) 
        recon_loss_avg = recon_loss_sum / num_views
 
        ########  TOTAL LOSS  ########
        total_loss = recon_loss_avg + beta * batch_averaged_regularizer  # scalar + scalar * scalar

        return total_loss, recon_losses, batch_averaged_regularizer
    
    def custom_kl(self, mean_0, var_0, mean_1, var_1):
        return - 0.5 * (
                torch.sum(
                    1 - var_0 / var_1
                    - (mean_0 - mean_1).pow(2) / var_1
                    + torch.log(var_0)
                    - torch.log(var_1),
                    dim=-1,
                )
            )