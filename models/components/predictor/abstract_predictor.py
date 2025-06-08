# a predictor consumes latent codes (or a derivative thereof) and returns a severity score.

import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractPredictor(ABC, nn.Module):
    # returns a distribution over classes (and not the index/numerical value of the predicted class)
    @abstractmethod
    def forward(self, means, std_devs):
        pass

    @abstractmethod
    def train_component(self, model, severity_train_loader, train_data_logger, validation_data_logger, train_data_evaluator, validation_data_evaluator, save_dir):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass

    # does not return a loaded model but loads the data into self
    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_summary(self):
        pass
