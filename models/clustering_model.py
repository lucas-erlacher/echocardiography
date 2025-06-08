# use baseline extractor for feature extraction. then embed all training data points int the learnt latent space 
# (and optionally reduce the dimensionsionality of this set). at inference time simply use k-nearest-neightbours to assign a class to new patient.

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from helpers import *
from models.abstract_model import AbstractModel
from models.components.extractor.baseline_extractor import BaselineExtractor
from models.components.predictor.clustering_predictor import ClusteringPredictor

class ClusteringModel(AbstractModel):
    def __init__(self, config, train_ids_path):
        self.extractor = BaselineExtractor(config)
        self.predictor = ClusteringPredictor(config)
        # complete the model setup 
        super().__init__(train_ids_path)