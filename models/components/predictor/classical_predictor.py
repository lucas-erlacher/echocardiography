# OUTDATED: old code that has not been tested with recent changes to other parts of the code i.e. if time fix it (or delete it)
# - logistic regression mode was fixed in February 2025 but all other modes have not been tested since October 2024 (i.e. they could be broken)

# predictor that utilizes "classical" (in the sense that they are not neural network based) ML methods 

import os
import torch
import numpy as np
import pickle
from models.components.predictor.abstract_predictor import AbstractPredictor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  
from helpers import *
from tqdm import tqdm
import torch.nn as nn
from models.components.component_helpers import *

class ClassicalPredictor(AbstractPredictor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.method = self.config["classical_method"]
        self.model = None
        if self.method == 'logistic_regression':
            self.model = LogisticRegression(max_iter=self.config["logistic_regression_max_iter"])
        elif self.method == 'svm':
            self.model = SVC(kernel=self.config["svm_kernel"], probability=True)
        elif self.method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=self.config["random_forest_n_estimators"])
        elif self.method == 'xgboost':
            obj = "binary:logistic" if self.config["binary_mode"] else "multi:softprob"
            self.model = XGBClassifier(n_estimators=self.config["xgboost_n_estimators"], objective=obj)

    def forward(self, means, std_devs):
        new_data = np.concatenate([means.cpu().detach().numpy(), std_devs.cpu().detach().numpy()], axis=1)
        pred_probs = self.model.predict_proba(new_data)
        distrib = to_distrib(len(pred_probs), torch.tensor(np.argmax(pred_probs, axis=1), device=self.config["device"]), self.config["binary_mode"])
        return distrib

    def train_component(self, model, cross_val, severity_train_loader, train_data_logger, validation_data_logger,  train_data_evaluator, validation_data_evaluator, save_dir):
        points = []
        classes = []
        for _, batch in enumerate(tqdm(severity_train_loader)):
            clips, labels = batch
            clips = [tensor.to(self.config["device"]) for tensor in clips]
            concatenated_embeddings, concatenated_std_devs = model.extractor.encode(clips)
            points.append(np.concatenate([concatenated_embeddings.cpu().detach().numpy(), concatenated_std_devs.cpu().detach().numpy()], axis=1))
            classes.append(labels.cpu().detach().numpy())
            del clips, labels, concatenated_embeddings, concatenated_std_devs, batch  # prevent out of memory issues
        
        points = np.concatenate(points)
        classes = np.concatenate(classes)
        self.model.fit(points, classes)
        self.save(save_dir)  # trainer assumes that after train_component returns a model has been saved (s.t. it can load and eval that model)

    def save(self, save_dir):
        model_filename = os.path.join(save_dir, 'predictor.pkl')
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load(self, save_dir):
        model_filename = os.path.join(save_dir, 'predictor.pkl')
        with open(model_filename, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def get_summary(self):
        s = ""
        s += "classifier method: " + self.method + "\n"
        if self.method == 'logistic_regression':
            s += "logistic_regression_max_iter: " + str(self.config["logistic_regression_max_iter"]) + "\n"
        elif self.method == 'svm':
            s += "svm_kernel: " + self.config["svm_kernel"] + "\n"
        elif self.method == 'random_forest':
            s += "random_forest_n_estimators: " + str(self.config["random_forest_n_estimators"]) + "\n"
        elif self.method == 'xgboost':
            s += "xgboost_n_estimators: " + str(self.config["xgboost_n_estimators"]) + "\n"
        return s