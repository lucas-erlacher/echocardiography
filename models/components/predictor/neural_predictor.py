# feed forward network based predictor

import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch.nn as nn
from tqdm import tqdm
from helpers import * 
from models.components.predictor.abstract_predictor import AbstractPredictor
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.components.component_helpers import *

class NeuralPredictor(AbstractPredictor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        temporal_parameter_factor = self.config["temporal_parameter_factor"] if (self.config["temporal_mode"] == "channel") else 1

        self.act_funct = nn.ReLU()  # ReLU seems to be a decent first choice in the abscence of better ideas/problems necessitating a different funct
        self.dropout = nn.Dropout(p=self.config["dropout"])
        self.num_classes = 2 if self.config["binary_mode"] == True else 3
        self.fc_1 = self.__get_orthogonal_linear(self.config["num_echo_views"] * 2 * self.config["mean_vector_dim"], 32 * temporal_parameter_factor)
        self.fc_2 = self.__get_orthogonal_linear(32 * temporal_parameter_factor, 8 * temporal_parameter_factor)
        self.fc_3 = self.__get_orthogonal_linear(8 * temporal_parameter_factor, self.num_classes)
        self.layers = nn.ModuleList([self.fc_1, self.fc_2, self.fc_3])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(layer.out_features) for layer in self.layers])
        self.softmax = nn.Softmax(dim=1)  # softmax to turn last 4 activations into probability distribution

    def forward(self, means, std_devs):
        combination = torch.cat((means, std_devs), dim=1)
        curr_activation = combination
        for i, layer in enumerate(self.layers):
            curr_activation = layer(curr_activation)
            curr_activation = self.batch_norms[i](curr_activation)
            curr_activation = self.act_funct(curr_activation)
            if i % self.config["dropout_freq"] == 0: 
                curr_activation = self.dropout(curr_activation)
        # NOTE: currently I am trainig predictor using cross entropy and that acutally expects an unnormalized distrib (i.e. don't apply softmax) 
        # distrib = self.softmax(curr_activation)
        return curr_activation
    
    # cross_val switches between cross_validation and full_data_training mode
    def train_component(self, full_model, cross_val, loader, predictor_train_data_logger, extractor_train_data_logger, train_data_evaluator, validation_data_evaluator, save_dir):
        # unpack extractor and predictor
        model = full_model.predictor.to(self.config["device"])
        extractor = full_model.extractor.to(self.config["device"])
        # disable gradient computation for extractor in case we are not fine tuning it
        if self.config["tune_extractor"] == False:
            for param in extractor.parameters(): param.requires_grad = False

        # training setup
        criterion = self.__get_criterion()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr_predictor"], weight_decay=self.config["weight_decay"])
        extractor_optimizer = torch.optim.Adam(extractor.parameters())  # if extractor is being tuned do so with standard lr
        if self.config["use_cos_annealing"]: scheduler = CosineAnnealingLR(optimizer, T_max=self.config["cos_annealing_t_max"], eta_min=self.config["cos_annealing_eta_min"])

        # training loop
        best_metrics = [(- sys.maxsize) for _ in range(2)]  # remember best metrics on eval data 
        for epoch_num in range(self.config["num_epochs_predictor"]):
            for batch_idx, batch in enumerate(tqdm(loader)):
                self.__passes(batch, extractor, model, criterion)
                # gradient logging (has to happen before we call zero_grad)
                if cross_val and (epoch_num % self.config["evaluation_freq_predictor"] == 0) and batch_idx == 0:
                    # it's fitting to log grad hists to train_data tb as these plots are generated using train data
                    # predictor grads
                    predictor_train_data_logger.log_gradients(model)  
                    # only log extractor grads if we are actually fine tuning it
                    models = []
                    for vae in extractor.unimodal_extractors.values():
                        l = [vae.encoder, vae.decoder]  # one enc - dec pair per view
                        models.append(l)
                    extractor_train_data_logger.log_gradients(models)
                    extractor_train_data_logger.increment_step()
                    del models
                self.__step(optimizer, extractor_optimizer, epoch_num)
                # more logging 
                if cross_val and (epoch_num % self.config["evaluation_freq_predictor"] == 0) and batch_idx == 0:
                    # it is interesting to see eval metrics (loss components, latent space visualizations, etc.) on both train and validation data
                    _, _ = train_data_evaluator.predictor_eval(extractor, model, criterion)  # not interested in train data metrics (but call still logs data to tb)
                    _, balanced_accuracy = validation_data_evaluator.predictor_eval(extractor, model, criterion)
                    print("curr_acc: " + str(balanced_accuracy))
                    del balanced_accuracy
            if check_exit(): return
            if self.config["use_cos_annealing"]: scheduler.step()
        
    def tune(self, num_epochs, full_model, loader, mode, lr):
        # unpack extractor and predictor
        model = full_model.predictor.to(self.config["device"])
        extractor = full_model.extractor.to(self.config["device"])

        ############    PARAMETER FREEZING    ############
        # deactivating all non-bn params resulted in an error -
        # therefore we now first deactivate all params just to then re-activate the batchnorm params that we want to tune
        if mode == "first-batchnorm":  
            # only tune first bn layer of extractor
            self.__deactivate_all(model)
            self.__deactivate_all(extractor)
            # reactivate first batchnorm layer of extractor
            for name, layer in extractor.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d) and name.split(".")[-1] == "bn_1":  
                    for param in layer.parameters(): param.requires_grad = True
        elif mode == "all-batchnorm":  
            # tune all bn layers of extractor and predictor
            self.__deactivate_all(model)
            self.__deactivate_all(extractor)
            self.__activate_batchnorm_layers(model)
            self.__activate_batchnorm_layers(extractor)
        elif mode == "all-params-extractor":
            # deactivate all predictor parameters and keep extractor parameters as they are
            self.__deactivate_all(model)
        elif mode == "all-params":
            # "all_params" mode tunes all parameters i.e. in that case don't freeze (or unfreeze) anything
            pass
        elif mode == "final-layer-predictor":
            self.__deactivate_all(extractor)
            self.__deactivate_all(model)
            # reactive params of final predictor layer
            for name, module in model.named_modules():
                if name == "fc_3":
                    for param in module.parameters(): param.requires_grad = True

        # training setup
        criterion = self.__get_criterion()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        extractor_optimizer = torch.optim.Adam(extractor.parameters(), lr=lr)
        
        # training loop
        for epoch_num in range(num_epochs):
            for _, batch in enumerate(tqdm(loader)):
                self.__passes(batch, extractor, model, criterion)
                self.__step(optimizer, extractor_optimizer, epoch_num)

    def __deactivate_all(self, model):
        for param in model.parameters(): param.requires_grad = False

    def __activate_batchnorm_layers(self, model):
        for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):    
                    for param in layer.parameters(): param.requires_grad = True

    def __get_criterion(self):
        w_zero = self.config["weight_zero"]
        weigths = None
        if self.num_classes == 2:
            weights = [w_zero, 1 - w_zero]
        elif self.num_classes == 3:
            weights = [w_zero, (1 - w_zero) / 2, (1 - w_zero) / 2]
        weights_tensor = torch.tensor(weights).to(self.config["device"])
        criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=self.config["label_smoothing"])
        return criterion
        
    # performs forward and backward passes 
    def __passes(self, batch, extractor, model, criterion):
        clips, labels = batch
        labels = labels.clone().detach()
        clips = [tensor.to(self.config["device"]) for tensor in clips]
        concatenated_embeddings, concatenated_std_devs = extractor.encode(clips, sample=self.config["implicit_data_augmentation"])
        pred = model(concatenated_embeddings, concatenated_std_devs)
        loss = criterion(pred, labels.to(self.config["device"]).long())
        loss.backward()
        
    # performs optmizer step
    def __step(self, optimizer, extractor_optimizer, epoch_num):
        ########  ALTERNATING MODE  ########
        # n steps of predictor optimization, n steps of extractor optimization, n steps of predictor optimization, etc. etc.
        if self.config["alternate_steps"] != -1:
            if (epoch_num // self.config["alternate_steps"]) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()
            else:
                extractor_optimizer.step()
                extractor_optimizer.zero_grad()

        ########  NON-ALTERNATING MODE  ########
        else: 
            if self.config["tune_extractor"] == False:
                # if we are not fine tuning the extractor just make steps on the predictor
                optimizer.step()
                optimizer.zero_grad()
            else:
                # if we are fine tuning extractor then in any epoch make steps on extractor and once the warmup phase is over also on predictor
                extractor_optimizer.step()
                extractor_optimizer.zero_grad()
                if epoch_num > self.config["extractor_warmup"]:
                    optimizer.step()
                    optimizer.zero_grad()

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir + '/predictor.pth')

    def load(self, path):
        load_weights(path + '/predictor.pth', self, self.config["device"])  # load weights at path into current instance

    def get_summary(self):
        s = ""
        s += "lr_predictor: " + str(self.config["lr_predictor"]) + "\n"
        s += "use_cos_annealing: " + str(self.config["use_cos_annealing"]) + "\n"
        s += "cos_annealing_t_max: " + str(self.config["cos_annealing_t_max"]) + "\n"
        s += "cos_annealing_eta_min: " + str(self.config["cos_annealing_eta_min"]) + "\n"
        s += "alternate_steps: " + str(self.config["alternate_steps"]) + "\n"
        s += "tune_extractor: " + str(self.config["tune_extractor"]) + "\n"
        s += "extractor_warmup: " + str(self.config["extractor_warmup"]) + "\n"
        s += "implicit_data_augmentation: " + str(self.config["implicit_data_augmentation"]) + "\n"
        s += "weight_zero: " + str(self.config["weight_zero"]) + "\n"
        s += "weight_decay: " + str(self.config["weight_decay"]) + "\n"
        s += "dropout: " + str(self.config["dropout"]) + "\n"
        s += "dropout_freq: " + str(self.config["dropout_freq"]) + "\n"
        s += "label_smoothing: " + str(self.config["label_smoothing"]) + "\n"
        return s
    
    def __get_orthogonal_linear(self, num_in, num_out):
        lin_layer = nn.Linear(num_in, num_out)
        torch.nn.init.orthogonal_(lin_layer.weight, 0.5)
        return lin_layer