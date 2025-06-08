import os
import sys
import torch
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from helpers import *
import random
import numpy as np
import torch
from models.baseline import Baseline

class Evaluator:
    def __init__(self, config, logger, eval_loader):
        self.logger = logger
        self.eval_loader = eval_loader
        self.loader_length = len(self.eval_loader)  
        self.block_mode = get_block_mode(config)
        # only used in block_mode (to avoid out of memory errors) and even then only has effect if loader is long
        self.temporal_max = 100 if self.loader_length > 100 else self.loader_length  
        self.config = config
        self.final_thresholds = [0, 0.2, 0.4, 0.5, 0.6, 0.8]

    ############    EXTRACTOR    ############

    def extractor_eval(self, extractor, beta):
        extractor.eval()
        # log input and recon images of a random index
        image_index = random.randint(0, self.temporal_max - 1 if self.block_mode else (self.loader_length - 1))  
        data_per_view = [[[], [], [], [], []] for _ in self.config["view_names"]]  # 5 lists in the list since we have 5 types of info to store
        ins = [None for _ in range(self.config["num_echo_views"])]
        recons = [None for _ in range(self.config["num_echo_views"])]
        t_sne_latents = torch.empty((0, self.config["mean_vector_dim"] * self.config["num_echo_views"]))  # accumulate concatenated means
        t_sne_labels = torch.empty((0))
        with torch.no_grad():  # disable gradient calculations
            for eval_batch_idx, eval_batch in enumerate(self.eval_loader):                  
                eval_clips, eval_labels = eval_batch
                eval_clips = [tensor.to(self.config["device"]) for tensor in eval_clips]
                recon, random_frames, mean, std_dev = extractor(eval_clips)
                concatenated_means, _ = extractor.encode(eval_clips)
                loss_info = extractor.compute_loss(random_frames, beta, recon, mean, std_dev)     
                for view_idx, view_key in enumerate(self.config["view_names"]):
                    total_loss, recon_loss, reduced_kl_loss = loss_info[view_idx]
                    # store data in format that logger expects
                    data_per_view[view_idx][0].append(total_loss)
                    data_per_view[view_idx][1].append(recon_loss)
                    data_per_view[view_idx][2].append(reduced_kl_loss)
                    data_per_view[view_idx][3].append(beta)
                    data_per_view[view_idx][4].append((beta * reduced_kl_loss))
                # collect data for t-sne
                t_sne_latents = torch.cat((t_sne_latents, concatenated_means.to("cpu")), dim=0)
                t_sne_labels = torch.cat((t_sne_labels, eval_labels), dim=0)
                # for every view log input and respective recon image once per eval (for a random index)
                if eval_batch_idx == image_index:
                    ins = [random_frames[view_key][0].unsqueeze(0).unsqueeze(0) for view_key in self.config["view_names"]]
                    recons = [recon[view_key][0].unsqueeze(0).unsqueeze(0) for view_key in self.config["view_names"]]
                # dataset in block_mode is very large i.e. stop before end of loader to avoid memory issues
                if self.block_mode and eval_batch_idx > self.temporal_max:
                    break  
        extractor.train()
        self.logger.log_loss_info(["total loss", "recon loss", "kld loss", "beta", "weighted kld loss"],
                                   torch.tensor(data_per_view))
        self.logger.log_image(ins, card_title="input")
        self.logger.log_image(recons, card_title="recon")
        # visualize latent space of entire extractor (i.e. not of individual extractors) as that is what predictor sees/has to work with
        if self.config["visualization_limit"] != -1:
            self.logger.visualize_latent_space(t_sne_latents, t_sne_labels)
        self.logger.increment_step()

    ############    PREDICTOR    ############

    def predictor_eval(self, extractor, predictor, criterion):
        predictor.eval()
        true_labels = []
        predicted_labels = []
        probs = []
        errors = torch.empty((0))
        with torch.no_grad():
            for eval_batch_idx, eval_batch in enumerate(self.eval_loader):
                eval_clips, eval_labels = eval_batch
                eval_clips = [tensor.to(self.config["device"]) for tensor in eval_clips]
                eval_concatenated_embeddings, eval_concatenated_std_devs = extractor.encode(eval_clips)
                eval_pred = predictor(eval_concatenated_embeddings, eval_concatenated_std_devs)
                probs += eval_pred.tolist()
                _, max_inds = torch.max(eval_pred, dim=1)
                # collect predicted/true labels for confusion matrix
                true_labels += eval_labels.tolist()
                predicted_labels += max_inds.tolist()
                loss_value = criterion(eval_pred, eval_labels.to(self.config["device"]).long())
                # accumulate loss values into error tensor
                errors = torch.cat((errors, loss_value.unsqueeze(0).to("cpu")), dim=0)
                # dataset in block_mode is very large i.e. stop before end of loader to avoid memory issues
                if self.block_mode and eval_batch_idx > self.temporal_max:
                    break  
        predictor.train()
        self.logger.log_loss_info(["mean CE"], errors.unsqueeze(0))  # unsqueeze to add "info_types" dimension that logger will iterate over
        self.logger.log_confusion_matrix(true_labels, predicted_labels)
        self.logger.increment_step()
        # compute and return metrics
        metrics = get_metrics(np.array(true_labels), np.array(predicted_labels), np.array(probs), self.config)
        auroc = metrics["auroc"]
        balanced_accuracy =  metrics["balanced_accuracy"]
        return auroc, balanced_accuracy
    
    ############    FINAL EVALUATION    ############

    # evaluate passed extractor-predictor pair on dataloader that was passed into constructor and reutrn metrics on the obtained predictions.
    # if subset is set we assume that dataset only contains those views and fill all missing views with zero_tensors.
    def final_evaluation(self, model, present_views=None, fine_tuning=False):
        model.extractor.eval()
        model.predictor.eval()
        metrics_list = []
        thresholds = []
        threshold_options = self.final_thresholds
        if fine_tuning: threshold_options = [0.5]  # if we are evaluating a fine tuned model we do not need to consider all ts but only 0.5
        with torch.no_grad():
            # run patient level inference with all threshold(s) unless we are in block_mode (since models that trigger block_mode do not use thresholds)
            if self.block_mode:
                true_labels, predicted_labels, probs, all_frame_preds, patient_stats = self.collect_predictions(model, [], present_views)
                metrics = get_metrics(np.array(true_labels), np.array(predicted_labels), np.array(probs), self.config)
                metrics_list.append(metrics)
            else:
                if self.config["binary_mode"]:
                    for t in threshold_options: 
                        thresholds.append([t])
                        metrics = self.__evaluate_with_threshold(model, [t], present_views)
                        metrics_list.append(metrics)
                else:
                    for t_1 in threshold_options: 
                        for t_2 in threshold_options: 
                            thresholds.append([t_1, t_2])
                            metrics = self.__evaluate_with_threshold(model, [t_1, t_2], present_views)
                            metrics_list.append(metrics)
        return metrics_list
    
    # ASSUMES: batch_size = 1 
    def __evaluate_with_threshold(self, model, ts, present_views):
        true_labels, predicted_labels, probs, all_frame_preds, patient_stats = self.collect_predictions(model, ts, present_views)
        # compute and return metrics
        metrics = get_metrics(np.array(true_labels), np.array(predicted_labels), np.array(probs), self.config)
        return metrics
    
    def collect_predictions(self, model, ts, present_views=None):
        true_labels = []
        predicted_labels = []
        probs = []
        all_frame_preds = []
        patient_stats = []
        for i, eval_batch in enumerate(self.eval_loader):
            eval_clips, eval_labels = eval_batch
            assert eval_labels.shape[0] == 1  # assert assumption
            patient_stats.append(get_patient_stats(eval_clips, i))

            # zero padding
            if present_views != None:
                padded_clips = []
                pad_shape = eval_clips[0]
                for view in self.config["view_names"]:
                    if view in present_views: 
                        next_eval_clip = eval_clips.pop(0)
                        padded_clips.append(next_eval_clip)
                    else:
                        padded_clips.append(torch.zeros_like(pad_shape))
                eval_clips = padded_clips

            eval_clips = [tensor.to(self.config["device"]) for tensor in eval_clips]

            # use multiframe_squential mode in case it is supported (= by baseline and even then only if we in spatial_mode). 
            # else use the generic inference method. 
            if isinstance(model, Baseline) and self.config["temporal_mode"] == None: 
                # using multiframe_sequential mode with self.config["final_eval_samples"] means that
                # we are conducting our evaluation over all frames that final eval dataset gives us per patient.
                # previously we had used multiframe_sampling mode BUT we changed that to 
                # remove the influence of fortunate/unfortunate random sampling on final evaluation score. 
                eval_pred, frame_preds = model.generate(eval_clips, multiframe_sequential=self.config["final_eval_num_frames"], thresholds=ts)
                all_frame_preds.append(frame_preds)
                # model returns frame level preds which can be used to compute class level probabilities
                probs += self.__compute_probs(frame_preds)
            else:
                eval_pred = model.generate(eval_clips)
                # if model does not return frame level predictions compute class probabilities as softmax of model predictions 
                # (which model does not force into a valid distribution e.g. neural network based predictor returns unnormalized outputs of last layer).
                probs += torch.nn.Softmax(dim=1)(eval_pred).tolist()
            # compute probs from frame_preds
            _, max_inds = torch.max(eval_pred, dim=1)
            # collect predicted/true labels for confusion matrix
            true_labels += eval_labels.tolist()
            predicted_labels += max_inds.tolist()
        return true_labels, predicted_labels, probs, all_frame_preds, patient_stats
    
    # compute class probabilities as relative frequencies of each class in the frame level predictions
    def __compute_probs(self, frame_preds):
        classes = [0, 1]
        if self.config["binary_mode"] == False: classes.append(2)

        batched_freqs = []  # for all items in batch store the freqs of each class 
        for item in frame_preds:
            counts = []
            for c in classes:
                count = item.count(c)
                counts.append(count / len(item))
            batched_freqs.append(counts)

        return batched_freqs