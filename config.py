import torch
import os

def log_config(c, writers, datasets, model, beta_scheduler, cross_val, fold_idx, num_folds):
    config_string = ""
    config_string += "-------------------------------------------------------------- \n"
    config_string += "                            CONFIG \n"
    config_string += "-------------------------------------------------------------- \n\n"
    
    config_string += "GENERAL \n"
    config_string += "seed: " + str(c["seed"]) + "\n"
    config_string += "device: " + str(c["device"]) + "\n"
    config_string += "node name: " + str(c["node_name"]) + "\n"
    config_string += "binary mode: " + str(c["binary_mode"]) + "\n"
    config_string += "fine_binary: " + str(c["fine_binary"]) + "\n"
    config_string += "cross_val: " + str(cross_val) + "\n"
    config_string += "fold_idx: " + str(fold_idx) + "\n"
    config_string += "num_folds: " + str(num_folds) + "\n"
    config_string += "\n"
    
    config_string += "DATA \n"
    config_string += "view names: " + str(c['view_names']) + "\n"
    config_string += "batch size extractror: " + str(c['batch_size_extractor']) + "\n"
    config_string += "batch size predictor: " + str(c['batch_size_predictor']) + "\n"
    for dataset in datasets:
        config_string += dataset.get_distribution_summary() + "\n"
    config_string += "frame sidelength: " + str(c['frame_sidelen']) + "\n"
    config_string += "num_frames: " + str(c["num_frames"]) + "\n"
    config_string += "use_test_data: " + str(c["use_test_data"]) + "\n"
    config_string += "final_eval_num_frames: " + str(c["final_eval_num_frames"]) + "\n"
    config_string += "augmentation_intensity: " + str(c["augmentation_intensity"]) + "\n"
    config_string += "\n"
    
    config_string += "MODEL \n"
    config_string += "model name: " + c['model_name']  + "\n"
    config_string += "temporal_mode: " + str(c["temporal_mode"]) + "\n"
    config_string += "temporal_input_factor: " + str(c["temporal_input_factor"]) + "\n"
    config_string += "temporal_parameter_factor: " + str(c["temporal_parameter_factor"]) + "\n"
    config_string += "\n"
    config_string += model.get_summary()
    
    config_string += "TRAINING \n"
    config_string += "trained_extractor_path: " + str(c['trained_extractor_path']) + "\n"
    config_string += "trained_predictor_path: " + str(c['trained_predictor_path']) + "\n"
    config_string += "num_epochs_extractor: " + str(c['num_epochs_extractor']) + "\n"
    config_string += "num_epochs_predictor: " + str(c['num_epochs_predictor']) + "\n"
    config_string += "beta_scheduler: " + beta_scheduler.get_summary() + "\n"
    config_string += "evaluation_freq_extractor: " + str(c['evaluation_freq_extractor']) + "\n"
    config_string += "evaluation_freq_predictor: " + str(c['evaluation_freq_predictor']) + "\n"
    config_string += "visualization_limit: " + str(c['visualization_limit']) + "\n"
    config_string += "\n"
    
    config_string += "############################################################## \n"
    
    # print config to console to give user the chance to catch misconfigurations before executing a long training run 
    print(config_string)

    # log config_string to tbs of all passed writers to save config params alongside training results of the run
    for writer in writers:
        writer.add_text("config", config_string)

# paths
regensburg_wrapper_path = "/cluster/home/elucas/thesis/heart_echo/"

base_config = {
    "seed": 6,  # this seed results in a more or less equal class-distribution across the cross validation folds
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "node_name": os.popen("hostname").read().strip(),  # unfortunately pytorch requires the exact same hardware for reproducability i.e. we need to remember it
    "binary_mode": False,
    "fine_binary": False, # decides which binay problem is being trained if binary_model is ON (False trains "0 vs other" and True trains "1 vs 2") 

    # data
    "view_names": ["LA", "KAKL", "KAPAP", "KAAP", "CV"],  # ["LA", "KAKL", "KAPAP", "KAAP", "CV"]
    "batch_size_extractor": 32,  
    "batch_size_predictor": 256,  # experience has shown that predictor requires large batch_sizes (about 400 or larger) to learn anything
    "frame_sidelen": 128,  # power of two to give the convolutions neat dimensions to work on
    "num_frames": 1,  # how many frames of each video will be loaded by the dataset (except for final eval dataset which performs patient level inference)
                      # use 1 in order to use every frame of every patient in every epoch (would not be the case for num_frames > 1)
    "use_test_data": True,  # whether or not to use test or validation data in final evaluation
    "augmentation_intensity": 0.55,  # launched multiple runs each incrementing this param by .5 (and this value gave best validation performance)
    "loader_workers": 4, 
 
    # model
    "model_name": "baseline",  # ["baseline", "classical"]
    # the following parameters concern both extractor and predictor
    "temporal_mode": None,  # [None, "channel", "lstm"] where None enables spatial mode
    "temporal_parameter_factor": 2,  # scaling factor for hidden-layer-parameters of networks (keep this reasonable to avoid memory issues)
    # - extractor
    #     - baseline_extractor
            "lr_extractor": 1e-4,
            "mean_vector_dim": 32,
            "parameter_factor": 2,  # increases depth and parameter count in spatial_encoder and spatial_decoder
            "encoder": "spatial",  # ["spatial", "pretrained"]
            "vae_type": "mmvm",  # ["regular", "mmvm"]
                # mmvm parameters
                "alpha": 0.5,
    #       - spatial encoder
              "final_channel_depth": 16,  # channel depth of activation outputs of last conv layer of spatial-encoder
    #       - pretrained encoder
              "freeze_pretrained": False,
            # the following parameters concerns both channel and lstm temporal_mode
            "temporal_input_factor": 10,  # how many frames of temporal information to use 
    #       # the following parameters concern lstm temporal_mode
            "lstm_layers": 4,  # number of lstm cells that are stacked "on top of each other" i.e. are invoked back-to-back for each input token 
            "bidirectional_lstm": False,
    # - predictor
    #     - neural predictor 
            "lr_predictor": 1e-4,
            "use_cos_annealing": False,
            "cos_annealing_t_max": 10,
            "cos_annealing_eta_min": 1e-6,
            "alternate_steps": -1,  # set to -1 to use non-alternating mode
            "tune_extractor": True,
            "extractor_warmup": 0,  # idea here is to first for some epochs infuse gradients from supervised information into extractor before training predictor
            "implicit_data_augmentation": False,
            "weight_zero": (1/6),  # 1/3 would be default value for 3 class problem so let's halve the weight down to 1/6
            "weight_decay": 0,
            "dropout": 0,
            "dropout_freq": 2,  # apply dropout every dropout_freq-many layers
            "label_smoothing": 0,
    #     - classical_predictor
            "classical_method": "logistic_regression",  # ["logistic_regression", "svm", "random_forest", "xgboost"]
            "logistic_regression_max_iter": 1000,
            "svm_kernel": "poly",  # ["linear", "poly", "rbf", "sigmoid"]
            "random_forest_n_estimators": 100,
            "xgboost_n_estimators": 100,
    #     - clustering_predictor
            "clustering_method": "knn",
            "reduction_dim": None,
            "reduction_method": "pca",  # non-linear dimensionality reduction is supported too but it makes the inference proceess VERY slow
            "knn_k": 1,
            "gmm_n": 1,

    # training
    "trained_extractor_path": None,
    "trained_predictor_path": None,
    "num_epochs_extractor": 200,  
    "num_epochs_predictor": 0,  # this parameter has to be set very carefully as one epoch too much can already result in a significant amount of overfitting
    "beta_mode": "const",
    "beta_const_val": 1e-05,
    "evaluation_freq_extractor": 1,  # number of epochs between evals
    "evaluation_freq_predictor": 2,  # number of epochs between evals  
    "visualization_limit": -1,  # set to -1 to disable latent space visualization
}

######################################
########  DERIVED PARAMETERS  ########
######################################

# set the config entried that are derived from other config entries
base_config["num_echo_views"] = len(base_config["view_names"]) 

if base_config["temporal_mode"] == "channel": 
        base_config["mean_vector_dim"] = base_config["mean_vector_dim"] * base_config["temporal_parameter_factor"]
        base_config["final_channel_depth"] = base_config["final_channel_depth"] * base_config["temporal_parameter_factor"]

# the following two values are the lengths of the shortest clips of each video-folder.
# if we go above those values we will cause the dataset to discard the patient that has this shortest video (and we don't want to discard any patients).
# even though they look somewhat low both values should be large enough as a heartbeat spans on average 10-12 frames 
# i.e. we will always show the model at least one heartbeat per patient which should be enough information to make an informed prediction.
base_config["final_eval_num_frames"] = 125 if base_config["use_test_data"] else 13  

########################################################
########  MISCONFIGURATION WARNINGS/ASSERTIONS  ########
########################################################

# it is too easy to switch between binary and ternary mode while forgetting to adapt the weight_zero parameter i.e. 
# notify the user in case a non-standard weight_zero parameter is configured (in case this choice is intentional this warning can be ignored) 
if (base_config["binary_mode"] and base_config["weight_zero"] != (1/2)) or (not base_config["binary_mode"] and base_config["weight_zero"] != (1/3)): 
    print("WARNING: weight_zero not set to standard value of current mode. ensure that this choice is intentional.")

if base_config["fine_binary"]:
    assert base_config["binary_mode"], "implementation of fine_binary model assumes that binary_mode is ON"
