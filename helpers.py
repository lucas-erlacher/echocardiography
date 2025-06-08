import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import cv2
from datetime import datetime
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from models.baseline import Baseline
from models.clustering_model import ClusteringModel
from models.classical_model import ClassicalModel
from torchvision.transforms.v2 import GaussianNoise
from PIL import Image
from itertools import chain
from config import *
sys.path.append(os.path.abspath(regensburg_wrapper_path))
from heart_echo.pytorch.HeartEchoDataset import HeartEchoDataset
from heart_echo.Helpers.LabelSource import LABELTYPE
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import random

def contains_nan(array):
    if type(array) == torch.tensor:
        return torch.isnan(array).any()
    elif type(array) == np.ndarray:
        return np.isnan(array).any()

####################################
########    SEED SETTING    ########
####################################
# has to be called whenever a training run of some sort (e.g. model training or fine-tuning) is about to be performed (in order to ensure reproducability)
def set_seeds(config):
    random.seed(config["seed"])                
    np.random.seed(config["seed"])             
    torch.manual_seed(config["seed"])          
    torch.cuda.manual_seed(config["seed"])     
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)  # by default this is 4 which for some reason causes the training to freeze for batch sizes > 1 
    print("fixed seeds")  # confirm to user that seeds have been set

############################
########    DATA    ########
############################

# all patient ids whose label is nan, 0, 1 or 2 (= patients that can be used for recon task)
train_ids_all = list(chain(range(30, 32),  
                           range(33, 43), 
                           range(47, 51), 
                           range(53, 64), 
                           range(66, 76), 
                           range(78, 106), 
                           range(107, 142), 
                           range(142, 244)))
# all patient id's from the above set of id's that have nan as their PH severity label
train_ids_nan = [122, 103, 134, 42, 138, 140, 113, 90, 127, 241]  
# all patients that have at least one view missing
train_ids_missing_views = [31, 36, 37, 38, 47, 57, 58, 66, 68, 69, 71, 72, 74, 78, 95, 98, 104, 107, 130, 146, 159, 162, 183, 230, 242, 243]

def get_dataset_label():
    return LABELTYPE.PULMONARY_HYPERTENSION

def get_transforms(config):
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=(config["frame_sidelen"], config["frame_sidelen"]),
                                    interpolation=Image.BICUBIC),
                    transforms.Lambda(lambda x: histogram_eq(x)),
                    transforms.ToTensor()])
    augmented_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(size=(config["frame_sidelen"], config["frame_sidelen"]),
                                            interpolation=Image.BICUBIC),
                            transforms.RandomAffine(
                                degrees= 35 * config["augmentation_intensity"],
                                translate = (
                                    0.25 * config["augmentation_intensity"],
                                    0.25 * config["augmentation_intensity"]
                                )
                            ),
                            transforms.Lambda(lambda x: histogram_eq(x)),
                            transforms.RandomResizedCrop(size=(config["frame_sidelen"], config["frame_sidelen"]),
                                scale=(0.7, 1.2), ratio=(1, 1)),

                            transforms.ColorJitter(
                                brightness= 1.25 * config["augmentation_intensity"],     
                                contrast= 1.25 * config["augmentation_intensity"],       
                                saturation= 1.25 * config["augmentation_intensity"],     
                                hue=min(0.5, 1.25 * config["augmentation_intensity"])  # hue does not accept values larger than 0.5            
                            ),
                            # blurring 
                            transforms.RandomApply(
                                [transforms.GaussianBlur(kernel_size=(5, 5), sigma=(3.5, 3.5))], 
                                p=config["augmentation_intensity"]
                            ),
                            # some transforms need a tensor and not a PIL image
                            transforms.ToTensor(),
                            lambda x: salt_pepper_noise(x, thresh=0.05), 
                            GaussianNoise(
                                sigma = 0.35 * config["augmentation_intensity"]
                            )
                        ])
    return transform, augmented_transform

# we save ids that a model was trained on alnogside trained model i.e. can use this information to reconstruct list of ids that were left out for validaition  
def reconstruct_validation_ids(train_ids_path):
    with open(train_ids_path, "r") as f:
        train_ids = []
        lines = f.readlines()
        for line in lines:
            clean = line.replace(",", "").strip()
            train_ids.append(int(clean))
        complement = list(set(train_ids_all) - set(train_ids))
        return complement

# returns a dataset to be used for final evaluation.
# as the name suggest the ids passed to this method are supposed to be validation data. 
def get_validation_data_final_set(ids, config, view_names=None):
    transform, _ = get_transforms(config) 
    dataset_label = get_dataset_label()
    if view_names == None: 
        view_names = config["view_names"]
    return HeartEchoDataset(
            ids,
            get_block_mode(config),
            config["temporal_input_factor"],
            config["fine_binary"],
            view_names,
            scale_factor=0.25, 
            frame_block_size=config["final_eval_num_frames"],
            transform=transform, 
            label_type=dataset_label,
            balance=False,  # balancing (i.e. oversampling of less represented classes) does not have a benefit when doing evaluation
            binary_mode = config["binary_mode"],
            final_eval=True)

# this method returns either  
# - test set containing all patients OR 
# - tune set and test set (in that case provide split_number, a seed (used to perform the splitting) and a data transform (used to augment tune set))
def get_test_set(config, split_num=None, seed=None, view_names=None):
    ids = list(chain(range(1, 81)))  # as far as I can tell from "heart_echo/Helpers/video_angles_list_test.csv" no ids of test data have label nan
    if view_names == None:
        view_names = config["view_names"]
    if split_num == None:  # only return test set
        return __construct_test_set(config, ids, final_eval=True, view_names=view_names)
    else:  
        number_generator = random.Random(seed)
        train_ids = number_generator.sample(ids, split_num)  # randomly sample split_num many ids
        test_ids = list(filter(lambda x: not x in train_ids, ids))
        check_leakage(train_ids, test_ids)  # should never trigger but more assertions are never a bad thing
        tune_set = __construct_test_set(config, train_ids, final_eval=False, view_names=view_names)
        test_set = __construct_test_set(config, test_ids, final_eval=True, view_names=view_names)

        return tune_set, test_set

# final_eval indicates whether the dataset is intended for final evaluation or training
def __construct_test_set(config, ids, final_eval, view_names):
    regular_transform, augmented_transform = get_transforms(config)
    # infer dataset parameters based on final_eval
    block_size = None
    transform = None
    balance = None
    train = None
    if final_eval == True:
        block_size = config["final_eval_num_frames"]
        balance = False
        train = False
    else:
        block_size = config["num_frames"]
        balance = True
        train = True
    # construct and return dataset
    return HeartEchoDataset( 
            ids,
            get_block_mode(config),
            config["temporal_input_factor"],
            config["fine_binary"],
            view_names, 
            scale_factor=0.25, 
            videos_dir="/cluster/dataset/vogtlab/Projects/Heart_Echo/PH",  # for final eval we use the videos that are stored separately 
            cache_dir="/cluster/home/elucas/.heart_echo_test",  # if we don't set this cached np.arrays belonging to videos from "/cluster/dataset/vogtlab/Projects/Heart_Echo" will be loaded
            frame_block_size=block_size,
            transform=regular_transform, 
            label_type=get_dataset_label(),
            balance=balance,
            binary_mode = config["binary_mode"],
            test=True,
            final_eval=final_eval,
            train=train)  

def get_final_loader(config, dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=config["loader_workers"])

# assert that no patient in final_eval_dataset comes with multiple blocks (which would not correspond to patient level inference).
# in theory this property is already enforced by the dataset code but it can't hurt to re-check this (important) property.  
def assert_final_eval(final_eval_dataset):
    assert len(final_eval_dataset._patients) >= len(final_eval_dataset), "at least one patient has multiple blocks in dataset (which we don't want because we want to generate only one disease score prediction per patient)"

def check_leakage(val_ids, train_ids):
    section = list(filter(lambda x: x in val_ids, train_ids))
    if len(section) != 0: raise Exception("data leakage detected. \n val_ids: \n" + str(val_ids) + "\n train_ids: \n" + str(train_ids) + "\n intersection: \n" + str(section))

def histogram_eq(img):
    img_np = np.array(img)
    return Image.fromarray(cv2.equalizeHist(img_np))

def salt_pepper_noise(sample, thresh):
    noise = torch.rand(sample.shape)
    sample[noise < thresh] = 0
    sample[noise > 1 - thresh] = 1
    return sample

# some model configurations require the dataset to return single frames and others require blocks of frames.
# this method abstracts away the decision if dataset should be in block_mode or not (based on the provided configuration)
def get_block_mode(config):
    if config["model_name"] == "baseline" and config["temporal_mode"] != None:
        return True  
    else: 
        # all other configurations operate on single frames (and not blocks of frames)
        return False



#############################
########    MODEL    ########
#############################

def get_fresh_model(config, train_ids_path):
    model = None
    if config["model_name"] == "baseline": model = Baseline(config, train_ids_path)
    if config["model_name"] == "clustering": model = ClusteringModel(config, train_ids_path)
    if config["model_name"] == "classical": model = ClassicalModel(config, train_ids_path)
    return model




##############################
########    WRITER    ########
##############################

# create two writers, point them to unique log-dir for this training run and log config file to both tb's.
# we need two writers since we want to log metrics for on both evaluation and training data during training.
def create_writers(curr_dir):
    # create writers such that they both have their own events.out file that they can log to
    train_data_writer = SummaryWriter(curr_dir + "/train_data")
    validation_data_writer = SummaryWriter(curr_dir + "/validation_data")
    return train_data_writer, validation_data_writer




##################################
########    STATISTICS    ########
##################################

# ASSUMES: batch_size = 1
def get_patient_stats(clips, id):
    strings = []
    numbers = []
    # get all stats once more but now over all views
    for view_idx in range(len(clips)):
        clip = clips[view_idx]
        assert clip.shape[0] == 1  # assert assumption
        clip = clip.squeeze(0)  # since we assume batch_size = 1 we can safely squeeze it away here
        data = __get_clip_stats(clip, str(id) + "_" + str(view_idx))
        numbers.append(data)
        strings.append(__format_stats(data))
    # compute average stats over all views
    averages = []
    for stat_idx in range(len(numbers[0])):
        # collect curr_stat from all views
        l = []
        for view_idx in range(len(numbers)):
            l.append(numbers[view_idx][stat_idx])
        averages.append(sum(l) / len(l))
    strings.append("-" * 8)
    strings.append(__format_stats(averages))
    return strings

def __get_clip_stats(clip, name):
    assert len(clip.shape) == 3
    
    mean_intensity = round(torch.mean(clip).item(), 2)
    std_intensity = round(torch.std(clip).item(), 2)
    min_intensity = round(torch.min(clip).item(), 2)
    max_intensity = round(torch.max(clip).item(), 2)
    median_intensity = round(torch.median(clip).item(), 2)
    percentile_50 = round(torch.quantile(clip, 0.50).item(), 2)
    percentile_80 = round(torch.quantile(clip, 0.80).item(), 2)
    percentile_90 = round(torch.quantile(clip, 0.90).item(), 2)
    percentile_95 = round(torch.quantile(clip, 0.95).item(), 2)

    # save first frame to visually inspect them
    first_frame = clip[0]
    image = ToPILImage()(first_frame)
    save_path = os.path.join("/cluster/home/elucas/thesis/examples", name + ".png")
    image.save(save_path)
    
    return [mean_intensity, std_intensity, min_intensity, max_intensity, median_intensity, percentile_50, percentile_80, percentile_90, percentile_95]

def __format_stats(data):
    mean_intensity, std_intensity, min_intensity, max_intensity, median_intensity, percentile_50, percentile_80, percentile_90, percentile_95 = data
    return (
        f"mean: {mean_intensity:>8.2f} | std: {std_intensity:>8.2f} | min: {min_intensity:>8.2f} | "
        f"max: {max_intensity:>8.2f} | median: {median_intensity:>8.2f} | pctl_50: {percentile_50:>8.2f} | "
        f"pctl_80: {percentile_80:>8.2f} | pctl_90: {percentile_90:>8.2f} | pctl_95: {percentile_95:>8.2f}")




###############################
########    METRICS    ########
###############################

# expects inputs to have a batch_dim
def __get_auroc(true_labels, probs, config):
    # weird but the sklearn function exepects quite inconsistent inputs in binary and mutliclass mode ...
    if config["binary_mode"]:  
        return roc_auc_score(true_labels, probs[:, 1], average='weighted')
    else:
        # in forward method of predictor we do NOT normalize the returned class scores (because that is what the CE loss wants)
        # but the auroc method in multiclass mode does expect that they sum to 1 (so we ensure that via softmax)
        normed_probs = torch.softmax(torch.from_numpy(probs), dim=1).numpy()
        return roc_auc_score(true_labels, normed_probs, multi_class='ovo', average='weighted')  # use weighted as we have class imbalance in case
                                                                                                # balance in severity_val_dataset is set to False

def __get_f1(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels, average='weighted')

def __get_precision(true_labels, pred_labels):
    return precision_score(true_labels, pred_labels, average='weighted')

def __get_recall(true_labels, pred_labels):
    return recall_score(true_labels, pred_labels, average='weighted')
                                                                                    
def get_balanced_accuracy(true_labels, pred_labels):
    return balanced_accuracy_score(true_labels, pred_labels)

def get_metrics(true_labels, pred_labels, probs, config):
    return {
        "auroc": __get_auroc(true_labels, probs, config),
        "f1": __get_f1(true_labels, pred_labels),
        "precision": __get_precision(true_labels, pred_labels),
        "recall": __get_recall(true_labels, pred_labels),
        "balanced_accuracy": get_balanced_accuracy(true_labels, pred_labels),
    }

def write_metrics(metrics_list, path):
    with open(path, "w") as f:
        # header
        f.write(f"{'auroc':<20} {'f1':<20} {'precision':<20} {'recall':<20} {'balanced accuracy':<20}\n")
        # lines
        for metrics in metrics_list:
            auroc = f"{metrics['auroc']:.4f},"
            f1 = f"{metrics['f1']:.4f},"
            precision = f"{metrics['precision']:.4f},"
            recall = f"{metrics['recall']:.4f},"
            balanced_accuracy = f"{metrics['balanced_accuracy']:.4f},"
            f.write(f"{auroc:<20} {f1:<20} {precision:<20} {recall:<20} {balanced_accuracy}\n")