# ARCHITECTURE: 
# - encoder/decoder CNNs employ 2D convolutions i.e. we operate on images and not videos
# - to that end, we select a random frame from the incoming ECHO video (and not a specific frame e.g. systole)
# these two design decisions (using 2D convs (rather than 3D convs) and extracting random frames (rather than extracting specific frames e.g. systole)) 
# are both motivated by the fact they maximise the amount of training that we can use to train the reconstruction task 
# (using 3D convs (and therefore training on videos)/using only specific frames (e.g. systole) would both significantly reduce the amount of training data)
# which is something that we need to actively need to look after as we only have relatively few ECHO videos to train on. 

# TRAINING: 
# - first train VAEs (with ELBO) on frames from both datasets (Regensburg + Stanford)
# - then use that data to train MLP (e.g. with CE)

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from helpers import *
from models.components.extractor.baseline_extractor import BaselineExtractor
from models.components.predictor.neural_predictor import NeuralPredictor
from models.abstract_model import AbstractModel

class Baseline(AbstractModel):
    def __init__(self, config, train_ids_path):
        self.config = config
        self.extractor = BaselineExtractor(config)
        self.predictor = NeuralPredictor(config)
        # complete the model setup 
        super().__init__(train_ids_path)