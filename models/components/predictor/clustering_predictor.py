# OUTDATED: old code that has not been tested with recent changes to other parts of the code i.e. if time fix it (or delete it)

# clustering based predictor 
# 
# in the context of this class "training" is simply the collection and saving of the embeddings and labels of all training data points.
# the actual fitting of the clustering algo happens in forward since in case dimensionality reduction is used 
# a new fit has to be computed for each unseen datapoint (since the output of the dimnesionality reduction depends on the unseen datapoint)
# UPDATE: fitting a fresh classifier for each unseen datapoint would be the correct way to do it 
# but that has proven to be immensely computationally expensive so we will make a compromise and fit a fresh classifier per batch

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch
import numpy as np
import pickle
from models.components.predictor.abstract_predictor import AbstractPredictor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from helpers import *
from tqdm import tqdm
import torch.nn as nn
from reducer import Reducer
from models.components.component_helpers import *

class ClusteringPredictor(AbstractPredictor, nn.Module):
    # method has to be in ["knn", "gmm"]
    #
    # optionally one can enable dimnesionality reduction of the space in which clustering will occur by setting redution_dim.
    # in case this is done one also has to provide a string "reduction method" that corrsponds to (the sting identifier of) a method in the Reducer class. 
    def __init__(self, config):
        super().__init__()  # This is a nn.Module after all
        self.config = config
        self.method = self.config["clustering_method"]
        
        if self.method == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=self.config["knn_k"])
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=self.config["gmm_n"])

        self.reduction_dim = self.config["reduction_dim"]
        self.reduction_method = self.config["reduction_method"]
        if self.reduction_dim != None:
            self.red = Reducer(self.config["seed"], self.config["reduction_dim"])
            # assert that self.reduction method corresponds to a method in Reducer. otherwise we might run a long extractor training 
            # just to (once it is time to train the predictor) realize that reduction_method does not correspond to a method in Reducer. 
            if not hasattr(self.red, self.reduction_method): raise ValueError("invalid reduction method: " + self.reduction_method)

        # these will be set in train_component
        self.points = []
        self.labels = []

    def forward(self, x):
        new_datapoint = x.cpu().detach().numpy()  # sklearn needs the input to be np_array
        # append new datapoint to set of old datapoints
        all_points = np.concatenate((self.points, new_datapoint), axis=0)
        # apply dimensionality reduction if requested by client
        if self.reduction_dim != None:
            # need to dim reduction all points (train data points and new point) at once since output of dim reduction depends on the new point 
            reduction_fn = getattr(self.red, self.reduction_method)  # use reduction method specified by client in constructor. 
                                                                        # thanks to assertion in constructor the fetching of the metohd should not fail. 
            all_points = reduction_fn(all_points) 
        # fit the model using all (potenitally dim reduced) train data points
        self.model.fit(all_points[:len(self.points)], self.labels)  # fit clustering model to all (potentially dim reduced) train data points
        pred = self.model.predict(all_points[len(self.points):])  # apply clustering to all points in batch
        distrib = to_distrib(x.shape[0], torch.tensor(pred, device=self.config["device"]), self.config["binary_mode"]) 
        return distrib
        
    # only embeds training set into latent space and saves these latent embeddings (and their labels)
    def train_component(self, model, cross_val, severity_train_loader, train_data_logger, train_data_evaluator, validation_data_evaluator):
        points = []
        classes = []
        for _, batch in enumerate(tqdm(severity_train_loader)):
            clips, labels = batch
            clips = [tensor.to(self.config["device"]) for tensor in clips]
            concatenated_embeddings, concatenated_std_devs = model.extractor.encode(clips)  # embed training data points
            points.append(concatenated_embeddings.cpu().detach().numpy(), concatenated_std_devs.cpu().detach().numpy())
            classes.append(labels.cpu().detach().numpy())
        self.points = np.concatenate(points)
        self.labels = np.concatenate(classes)
        
    # since we are re-fitting the sklearn objects (knn, gmm) every forward pass (i.e. there is no point in saving any one of these fitted classifiers)
    # the only thing that is worth saving are the embeddings and labels that were collected in train_component (those won't change between forward passes)
    def save(self, save_dir):
        points_filename = os.path.join(save_dir, 'predictor_points.pkl')
        with open(points_filename, 'wb') as points_file: pickle.dump(self.points, points_file)
        
        labels_filename = os.path.join(save_dir, 'predictor_labels.pkl')
        with open(labels_filename, 'wb') as labels_file: pickle.dump(self.labels, labels_file)

    def load(self, save_dir):
        points_filename = os.path.join(save_dir, 'predictor_points.pkl')
        with open(points_filename, 'rb') as points_file: self.points = pickle.load(points_file)

        labels_filename = os.path.join(save_dir, 'predictor_labels.pkl')
        with open(labels_filename, 'rb') as labels_file: self.labels = pickle.load(labels_file)

    def get_summary(self):
        s = ""
        s += "clustering method: " + str(self.method) + "\n"
        if self.method == 'knn':
            s += "num neighbours: " + str(self.model.n_neighbors) + "\n"
        elif self.method == 'gmm':
            s += "num components: " + str(self.model.n_components) + "\n"
        s += "reduction dim: " + str(self.reduction_dim) + "\n"
        s += "reduction method: " + str(self.reduction_method) + "\n"
        return s