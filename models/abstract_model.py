# defines what all models need to have (s.t. classes such as the Evaluator can assume certain methods and fields)

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from helpers import *
from abc import ABC, abstractmethod
import torch
from models.components.component_helpers import * 

class AbstractModel(ABC):
    
    def __init__(self, train_ids_path):
        # load and remember id's that model was trained on
        train_ids = []
        with open(train_ids_path, "r") as f:
            lines = f.readlines()
            for l in lines: train_ids.append(int(l.strip()))
        self.train_ids = train_ids

        # enforce that every model has the following fields
        if not hasattr(self, 'extractor'):
            raise NotImplementedError("subclasses must define an 'extractor' attribute.")
        if not hasattr(self, 'predictor'):
            raise NotImplementedError("subclasses must define a 'predictor' attribute.")

    # used to perform a forward pass in the model. 
    # takes in a set of videos (from different views). 
    #
    # there are 2 different modes for the forward pass:
    # - NORMAL MODE: take the clip, sample a frame from it, embed that sample and then invoke predictor to turn embedding into severity score
    # - MULTIFRAME SEQUENTIAL MODE: 
    #   - if mulitframe_sequential is not None but an integer n derive a severity score from the first n frames of 
    #     the clip and compute the final severity score of that clip by comparing the class rates to the provided threshold(s)
    #   - in multiframe_sequential mode a 1-hot distribution is returned i.e. DO NOT use a it in situations where the full distrib 
    #     is needed (e.g. during training where cross entropy wants to have the full distrib and not just information about the argmax). 
    def generate(self, x, multiframe_sequential=None, thresholds=None):
        if multiframe_sequential == None:  # we are in normal mode
            mean, std_dev = self.extractor.encode(x)
            return self.predictor(mean, std_dev)
        else:  
            batch_size = x[0].shape[0]  # x[0] is the first view 
            num_frames = multiframe_sequential
            # collect the frame level predictions
            preds_per_batch = [[] for _ in range(batch_size)]  # for each item (= patient) in batch collect a list of frame-level class predictions 
            for i in range(num_frames):
                # for each view present in x pick out the i-th frame (which will disable frame sampling in extractor)
                # for each view and for each batch_ind reduce the full clip to the i-th frame
                one_frame_clips = [view[:, i, :, :].unsqueeze(1) for view in x]  # this slicing op removes the frame_dimension (bc we only select one frame)
                                                                                    # so add that dim back using unsqueeze (since extractor expects a frame_dim)
                mean, std_dev = self.extractor.encode(one_frame_clips)  
                pred = self.predictor(mean, std_dev)
                _, max_inds = torch.max(pred, dim=1)  # collapse prediction to argmax
                for batch_ind, predicted_ind in enumerate(max_inds):
                    preds_per_batch[batch_ind].append(predicted_ind.item())
            # compute class rates
            rates_per_batch = []  # for each item (= patient) in batch collect rate of how many frames voted for each class
            num_classes = 2 if self.config["binary_mode"] else 3
            for b in range(batch_size):
                rates = []
                for c in range(num_classes):
                    count = preds_per_batch[b].count(float(c))
                    rates.append(count / num_frames)
                rates_per_batch.append(rates)
            # compare rates to thershold(s) to obtain final prediction
            patient_pred_per_batch = []
            for b in range(batch_size):
                pred = 0
                # decide between 0 and other (= 1 in binary case or 1-2 in ternary case)       
                if rates_per_batch[b][0] > thresholds[0]: pred = 0
                else: pred = 1
                # if ternary_mode (and class 0 was not picked before): decide between 1 and 2
                if not self.config["binary_mode"] and pred == 1:
                    if rates_per_batch[b][1] > thresholds[1]: pred = 1
                    else: pred = 2
                patient_pred_per_batch.append(pred)
            # unlike in many above steps (where we had to manually manage the batch_size) to_distrib is already batched i.e. takes care of batch_dim internally 
            one_hots = to_distrib(max_inds.shape[0], torch.tensor(patient_pred_per_batch, device=self.config["device"]), self.config["binary_mode"])
            return one_hots, preds_per_batch

    def get_summary(self):
        s = ""
        s += "EXTRACTOR \n" + self.extractor.get_summary() + "\n"
        s += "PREDICTOR \n" + self.predictor.get_summary() + "\n"
        return s