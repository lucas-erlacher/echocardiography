# code assumes that list of patient_ids passed to constructor contains at least 1 patient of every class (0, 1 or 2) group (this is asserted in constructor)
# (otherwise the balancing loop will run infinitely as it will not be able to sample a patient from the class that is not present)

import torch
from torch.utils import data
import numpy as np
from ..Helpers import Helpers, LABELTYPE, LabelSource
import random

class HeartEchoDataset(data.Dataset):

    VIDEO_TYPES = ["mp4", "avi"]

    # [TODO] clean-up parameter list:
    # - remove up all parameter that are just config parameters i.e. remove all the individual parameters, take in config instead
    #   and unpack the respective parameter from config inside of the constructor.
    # - it might be that some parameters are implied by others e.g. I think balance is true iff train is true 
    #   (in that case balance could be removed and inferred in the constructor based on the value of train)
    def __init__(self, patient_IDs, block_mode, temporal_input_factor, fine_binary, video_angles=None, cache_dir="~/.heart_echo",
                 videos_dir="/cluster/work/vogtlab/Projects/Heart_Echo",
                 frame_block_size=10, scale_factor=1.0, label_type=LABELTYPE.NONE, transform=None, balance=False, binary_mode=False,
                 test=False, procs=4, p_mixup=0, final_eval=False, train=False):
        
        # NOTE: edited
        self.ids = patient_IDs
        self.balance = balance
        self.binary_mode = binary_mode
        self.test = test  # load data from hold-out folder (or not) 
        self.p_mixup = p_mixup
        self.final_eval = final_eval  # true for datasets that are used for final evaluation (but e.g. not for ones that are used for train-time evaluation)
        self.fine_binary = fine_binary 
        self.block_mode = block_mode
        self.train = train  # true for datasets that are used for training or fine-tuning 

        # if temporal channel mode is enabled we return temporal_input_factor many frames for each call to get_item i.e. 
        # hardcode frame_block_size to temporal_input_factor regardless of what value the dataset is being constructed with
        if self.block_mode:
            temporal_input_factor > 1, "temporal_input_factor must be > 1 otherwise block_mode does not supply temporal information."
        if self.block_mode: frame_block_size = temporal_input_factor

        # NOTE: edited over

        if video_angles is not None:
            self._multimodal = True
        else:
            self._multimodal = False

        # Unimodal variables
        self._frame_blocks = []
        self._labels = []

        # Multimodal variables
        self._patients = dict()
        self._index_list = []
        self._video_angles = video_angles

        # Label source
        self._label_source = LabelSource(self.test)
        self._label_type = label_type

        # Other
        self._num_samples = 0
        self._is_video = frame_block_size > 1

        # Load videos
        all_videos = Helpers.load_all_videos(cache_dir, videos_dir, scale_factor, self.test, procs)

        # Unimodal processing
        if not self._multimodal:
            # Select videos corresponding to given patient IDs
            video_names = all_videos.keys()
            for id in patient_IDs:
                for v in video_names:
                    if str(id) in v:
                        # Split into frame blocks, dropping the rest
                        split_frames = HeartEchoDataset._split_array(all_videos[v], frame_block_size, self.block_mode, False)
                        self._frame_blocks += split_frames

                        # Add labels based on video and label type
                        if label_type in [LABELTYPE.VISIBLE_FAILURE_WEAK, LABELTYPE.VISIBLE_FAILURE_STRONG]:
                            curr_id, curr_view = Helpers.get_id_and_angle_from_name(v)
                            label_value = self._label_source.get_label_for_patient_and_view(curr_id, curr_view,
                                                                                            label_type)

                        elif label_type == LABELTYPE.VIEW:
                            curr_id, curr_view = Helpers.get_id_and_angle_from_name(v)
                            label_value = curr_view

                        elif label_type != LABELTYPE.NONE:
                            label_value = self._label_source.get_label_for_patient(id, label_type)

                        else:
                            label_value = 0

                        self._labels += [label_value for i in range(len(split_frames))]

            self._num_samples = len(self._frame_blocks)

        # Multimodal
        else:
            # Identify patients for which all required video angles are given

            # NOTE: edited
            # because we do NOT want to filter our patients that have missing angles since we want to impute angles for such patients further down
            # previous code: 
            # matching_patient_IDs = Helpers.filter_patients_by_video_angle(patient_IDs, video_angles, self.test)
            # new code: 
            if self.test == False:  # do NOT filter test data since that could artificially make our final performance numbers look better
                matching_patient_IDs = Helpers.filter_patients_by_video_angle(patient_IDs, video_angles, self.test)  
            else: matching_patient_IDs = patient_IDs

            curr_index = 0

            # For each matching patient, save frame blocks for each angle
            for id in matching_patient_IDs:
                label_value = self._label_source.get_label_for_patient(id, label_type)

                # fine_binary mode only concerns itself with classes 1 and 2 (which contains class 3)
                if self.fine_binary and not (label_value in [1.0, 2.0, 3.0]): continue

                self._patients[id] = dict()

                min_blocks = 1000000

                for angle in video_angles:
                    video_name = "{}{}".format(id, angle)

                    # NOTE: edited
                    try:
                        blocks = HeartEchoDataset._split_array(all_videos[video_name], frame_block_size, self.block_mode, False)
                        if len(blocks) == 0: print("WARNING: frame_block_size (" +  str(frame_block_size) + ") leads to no blocks being generated for patient " + str(id) + " in view " + str(angle) + " whose num_frames is " + str(len(all_videos[video_name])))
                    except KeyError:
                        # impute black frames in case of missing angle
                        num_frames = None
                        if not self.test:
                            num_frames = 122  # videos from "/cluster/dataset/vogtlab/Projects/Heart_Echo/" are about 122 on average
                        else:
                            num_frames = 270  # videos from "/cluster/dataset/vogtlab/Projects/Heart_Echo/PH" are about 270 frames on average
                        num_blocks = num_frames // frame_block_size
                        sidelen = 200  # does not matter as frames will be rescaled by the transform that is passed to the dataset constructor
                        blocks = torch.zeros((num_blocks, frame_block_size, sidelen, sidelen))
                    # NOTE: this was how it was done before I replaced it with the code above (old code was simply crashing in case of a missing angle)
                    # blocks = HeartEchoDataset._split_array(all_videos[video_name], frame_block_size, False)

                    if len(blocks) < min_blocks:
                        min_blocks = len(blocks)

                    self._patients[id][angle] = blocks

                # NOTE: edited
                # in final eval mode we only want one block per patient since we are doing patient level inference 
                # i.e. we do not want to generate multiple (potentially different) predictions per patient (which could happen if a patient had multiple blocks)
                if self.final_eval: min_blocks = 1  # this will force every clip of every patient to have one block (which is what we want in final_eval mode)

                # Ensure same number of blocks
                for angle in video_angles:
                    self._patients[id][angle] = self._patients[id][angle][:min_blocks]

                # Update index list
                for j in range(min_blocks):
                    # NOTE: edited
                    if self.binary_mode:
                        if self.fine_binary:
                            # fine_binary mode: merge 3 into 1, shift 2 into 1 and shift 1 into 0
                            self._index_list.append((id, j, 1.0 if label_value == 3.0 else label_value - 1))
                        else:
                            # coarse_binary mode: merge 3 and 2 into 1, keep 1 and 0 as they are 
                            self._index_list.append((id, j, 1.0 if (label_value == 3.0 or label_value == 2.0) else label_value))  
                    else:
                        # ternary mode: merge class 3 patients into class 2 (don't do any other edits to the labels)
                        self._index_list.append((id, j, 2.0 if label_value == 3.0 else label_value))  
                    
                    # NOTE: old code
                    # self._index_list.append((id, j, label_value))

                curr_index += min_blocks

            self._num_samples = len(self._index_list)

        # output transformation
        self.transform = transform
        
        # NOTE: edited
        # create collections of patient_ids for each class (0, 1, 2 and nan)
        l = [(tup[0], tup[2]) for tup in self._index_list]
        l = list(set(l))
        # filter out tuples with matching class label
        ids_nan = list(filter(lambda x: np.isnan(x[1]), l))  
        ids_zero = list(filter(lambda x: x[1] == 0.0, l))
        ids_one = list(filter(lambda x: x[1] == 1.0, l))
        ids_two = list(filter(lambda x: x[1] == 2.0, l))
        # rm class label as that information is now given by the list name
        self.ids_nan = list(map(lambda x: x[0], ids_nan))  
        self.ids_zero = list(map(lambda x: x[0], ids_zero))  
        self.ids_one = list(map(lambda x: x[0], ids_one))  
        self.ids_two = list(map(lambda x: x[0], ids_two))   
        # warn user of missing classes (if necessary)
        if len(self.ids_zero) == 0: print("WARNING: no patients with label 0 passed to dataset")
        if len(self.ids_one) == 0: print("WARNING: no patients with label 1 passed to dataset")
        if not self.binary_mode:
            if len(self.ids_two) == 0: print("WARNING: no patients with label 2 passed to dataset")
        # NOTE: edited over

        # Balancing
        self._selected_rows = np.arange(self._num_samples)

        # NOTE: edited (commented out the following block since we will do balancing in __getitem__)
        '''
        if balance:
            if self._multimodal:
                # Get a complete array of labels
                label_array = np.empty(0, dtype=int)

                for tup in self._index_list:
                    label_array = np.append(label_array, tup[2])

                positive_indices = label_array == 1
                negative_indices = label_array == 0

            else:
                positive_indices = self._labels == 1
                negative_indices = self._labels == 0

            # More negative than positive
            if np.sum(negative_indices) > np.sum(positive_indices):
                selected_negative_rows = self._selected_rows[negative_indices]
                selected_positive_rows = self._selected_rows[positive_indices]
                np.random.shuffle(selected_negative_rows)
                self._selected_rows = np.append(selected_positive_rows,
                                                selected_negative_rows[:len(selected_positive_rows)])
            else:
                selected_negative_rows = self._selected_rows[negative_indices]
                selected_positive_rows = self._selected_rows[positive_indices]
                np.random.shuffle(selected_positive_rows)
                self._selected_rows = np.append(selected_negative_rows,
                                                selected_positive_rows[:len(selected_negative_rows)])

            np.random.shuffle(self._selected_rows)
            self._num_samples = len(self._selected_rows)

            # Ensure that there are still samples (depending on the video selection, there could be no positives)
            if self._num_samples == 0:
                raise ValueError("No samples remaining!")
        '''

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        index = self._selected_rows[index]
        # NOTE: edited
        if not self._multimodal:
            if self.transform is not None:
                if self._is_video:
                    return torch.stack([self.transform(f).squeeze() for f in self._frame_blocks[index]]), self._labels[index]
                else:
                    return self.transform(self._frame_blocks[index]).squeeze(), self._labels[index]
            else:
                return self._frame_blocks[index], self._labels[index]
        else:
            index_info = self._index_list[index]
            label = index_info[2]

            # balancing
            if self.balance:
                values = [0, 1]  # class merging has already been done in constructor i.e. label 3 does not exist in self._index_list anymore
                if not self.binary_mode: values.append(2)
                # if list of ids passed to constructor contains patients with label nan make sure such patients can be returned from __getitem__
                if len(self.ids_nan) > 0: values.append(np.nan)  
                curr_class = np.random.choice(values, p = ([1 / len(values)] * len(values)))  # class from which we want to return a patient in this invocation of __getitem__ 
                while not self.__equals(label, curr_class): 
                    random_index = np.random.randint(0, (len(self._index_list) - 1))  
                    index_info = self._index_list[random_index]  # sample a new block-info tuple from index_list
                    label = index_info[2]
                assert self.__equals(curr_class, label), "incorrect label after balancing: " + str(curr_class) + ", " + str(label)  # ensure that balancing works

            datapoint = self._patients[index_info[0]]

            if self.p_mixup != 0:
                # decide if we will do mixup in this get_item invocation
                random_num = np.random.rand()
                if random_num < self.p_mixup:
                    # sample another image from the same class
                    new_sample_index_info = None
                    new_sample_label = -20  # make sure the enter the while loop at least once
                    while not self.__equals(new_sample_label, curr_class): 
                        random_index = np.random.randint(0, (len(self._index_list) - 1)) 
                        new_sample_index_info = self._index_list[random_index]  # sample a new block-info tuple from index_list
                        new_sample_label = index_info[2]
                    # interleave the existing datapoint and the new sample (for all angles that are being used)
                    for angle in self._video_angles:
                        old_block = datapoint[angle][index_info[1]]
                        new_block = self._patients[new_sample_index_info[0]][angle][new_sample_index_info[1]]
                        merged_block = [self.__mixup_merge(old_block[i], new_block[i]) for i in range(len(old_block))]  # merge all frames of blocks
                        datapoint[angle][index_info[1]] = merged_block

            state = torch.get_rng_state()  # save current torch_rng state
            _ = torch.rand(1)  # advance rng state (so that the saved state is different in next invocation of __getitem__)
            if self.transform is not None:
                if self._is_video:
                    l = []
                    for angle in self._video_angles:
                        stack_list = []
                        data = self.__temporal_preprocess(datapoint[angle][index_info[1]])

                        for f in data:
                            transformed = self.__apply_transform(self.transform, f, state)
                            squeezed = transformed.squeeze()
                            stack_list.append(squeezed)
                        stacked = torch.stack(stack_list)
                        l.append(stacked)
                    return l, label
                else:
                    l = []
                    for angle in self._video_angles:
                        data = self.__temporal_preprocess(datapoint[angle][index_info[1]])
                        transformed = self.__apply_transform(self.transform, data, state)
                        squeezed = transformed.squeeze()
                        l.append(squeezed)
                    return l, label
            else:
                l = []
                for angle in self._video_angles:
                    l.append(self.__temporal_preprocess(datapoint[angle][index_info[1]]))
                return l, label
        # NOTE: this is how it was before:
        '''
        if not self._multimodal:
            if self.transform is not None:
                if self._is_video:
                    return torch.stack([self.transform(f).squeeze() for f in self._frame_blocks[index]]), self._labels[index]
                else:
                    return self.transform(self._frame_blocks[index]).squeeze(), self._labels[index]
            else:
                return self._frame_blocks[index], self._labels[index]
        else:
            index_info = self._index_list[index]
            if self.transform is not None:
                if self._is_video:
                    return [torch.stack(
                        [self.transform(f).squeeze() for f in self._patients[index_info[0]][angle][index_info[1]]]) for
                               angle in self._video_angles], index_info[2]
                else:
                    return [self.transform(self._patients[index_info[0]][angle][index_info[1]]).squeeze() for angle in
                            self._video_angles], index_info[2]
            else:
                return [self._patients[index_info[0]][angle][index_info[1]] for angle in self._video_angles],
                       index_info[2]
        '''

    # compares two labels in [0, 1, 2, np.nan] while handeling the nan comparison properly
    def __equals(self, label, curr_class):
        if np.isnan(curr_class):  
            if np.isnan(label): return True
            else: return False
        else: 
            return label == curr_class

    def get_label_balance(self):
        if self._label_type in [LABELTYPE.PRETERM_BINARY, LABELTYPE.VISIBLE_FAILURE_WEAK,
                                LABELTYPE.VISIBLE_FAILURE_STRONG]:
            if self._multimodal:
                sum_total = 0

                for idx in self._selected_rows:
                    sum_total += self._index_list[idx][2]

                return sum_total / self._num_samples

            else:
                return np.sum(np.array(self._labels)[self._selected_rows]) / self._num_samples
        else:
            raise ValueError("Only supported for PRETERM_BINARY, VISIBLE_FAILURE_WEAK, VISIBLE_FAILURE_STRONG")

    @staticmethod
    def _split_array(array, block_size, block_mode, keep_remainder=True):
        if block_mode:  
            # in temporal channel mode append space blocks by less than frame_block_size (to extract as much training data as possible from clips)
            blocks = []
            for start_idx in range(0, len(array) - block_size + 1, 2):
                block = array[start_idx : (start_idx + block_size)]
                blocks.append(block)
            return blocks
        else:  
            # use old code in spatial mode
            num_full_blocks = len(array) // block_size
            ret_list = [array[i * block_size:(i + 1) * block_size] for i in range(num_full_blocks)] 

            if keep_remainder:
                ret_list.append(array[(num_full_blocks - 1) * block_size:])

            # Special case if block_size is 1
            if block_size == 1:
                ret_list = [x.reshape(x.shape[1], -1) for x in ret_list]

            return ret_list

    def is_multimodal(self):
        return self._multimodal
    
    # NOTE: edited
    def get_distribution_summary(self):
        s = (
            f"class distribution  >>  "
            f"0: {len(self.ids_zero):<5}  "
            f"1: {len(self.ids_one):<5}  "
            f"2: {len(self.ids_two):<5}  "
            f"nan: {len(self.ids_nan):<5}  "
        )
        s += f"balance: {self.balance}"
        return s

    # NOTE: edited
    # merge two frames (current implementation: cut both images in half and concatenate them next to each other in width-dimension)
    def __mixup_merge(self, old_frame, new_frame): 
        # frames have not been resized yet (so crop to largest shared area)
        min_x = min(old_frame.shape[0], new_frame.shape[0])
        min_y = min(old_frame.shape[1], new_frame.shape[1])
        cropped_old = old_frame[:min_x, :min_y] 
        cropped_new = new_frame[:min_x, :min_y] 
        stacked = np.concatenate((cropped_old[:, :min_y // 2], cropped_new[:, min_y // 2:]), axis=1)
        return stacked

    # NOTE: edited 
    # if in block_mode:
    # - rotate (with wrap around) list of frames in the list by a randomly sampled offset.
    #   since this is a data augmentation functionality we only apply rotation if the current dataset is meant for training (this includes fine-tuning).
    #   this data augmentation scheme is designed to simluate different degrees of misalignemet in the cardiac cycle (between the different angles of a patient)
    # - concatenate the frames of a block in the channel_dimension into one image. 
    #   doing this before the spatial data augmentation transforms are applied is crucial otherwise each frame would get shifted and rotated differently
    #   (but if they are stacked into one image with many channels the transforms will alter all channels in sync). 
    def __temporal_preprocess(self, frame_list):
        if self.block_mode and self.train:
            ####  CIRCULAR FRAME ROTATION  ####
            length = len(frame_list)
            # offset can be 0 i.e. no rotation. offset = length is not useful (so exclude it from sampling) as that would result in an equivalent frame_list
            offset = np.random.randint(0, length)
            new_index = [((old_index + offset) % length) for old_index in range(length)]  # rotated index of each frame 
            rotated = [None for _ in range(length)]
            for i, frame in enumerate(frame_list):
                rotated[new_index[i]] = frame
            return rotated
        else: 
            return frame_list

    def __apply_transform(self, transform, x, seed):
        # in block_mode ensure that all frames of the block are augmented in the same way by setting the same seed before each frame-augmentation
        if self.block_mode: 
            torch.set_rng_state(seed)
        return transform(x)
