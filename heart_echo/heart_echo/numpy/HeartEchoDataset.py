import numpy as np
from ..Helpers import Helpers, LABELTYPE, LabelSource
from ..Processing import VideoUtilities, ImageUtilities


class HeartEchoDataset:
    VIDEO_TYPES = ["mp4", "avi"]

    def __init__(self, patient_IDs, video_angles=None, cache_dir="~/.heart_echo",
                 videos_dir="/cluster/work/vogtlab/Projects/Heart_Echo",
                 frame_block_size=10, scale_factor=1.0, label_type=LABELTYPE.NONE, balance=False, procs=4, resize=None):

        # Balancing only supported for binary labels
        if balance and label_type not in [LABELTYPE.PRETERM_BINARY, LABELTYPE.VISIBLE_FAILURE_WEAK,
                                          LABELTYPE.VISIBLE_FAILURE_STRONG]:
            raise ValueError(
                "Balancing only supported for PRETERM_BINARY, VISIBLE_FAILURE_WEAK, VISIBLE_FAILURE_STRONG")

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

        # Resizing
        self._resize = resize

        # Label source
        self._label_source = LabelSource()
        self._label_type = label_type

        # Other
        self._num_samples = 0

        # Load videos
        all_videos = Helpers.load_all_videos(cache_dir, videos_dir, scale_factor, procs)

        # Unimodal processing
        if not self._multimodal:
            # Select videos corresponding to given patient IDs
            video_names = all_videos.keys()
            for id in patient_IDs:
                for v in video_names:
                    if str(id) in v:
                        # Split into frame blocks, dropping the rest
                        split_frames = HeartEchoDataset._split_array(all_videos[v], frame_block_size, False)
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

            # Do resizing
            if not self._multimodal:
                if frame_block_size > 1:
                    self._frame_blocks = [VideoUtilities.resize_video(frames, self._resize[0], self._resize[1]) for
                                          frames
                                          in
                                          self._frame_blocks]
                else:
                    self._frame_blocks = [ImageUtilities.resize_image(img_array, self._resize[0], self._resize[1]) for
                                          img_array in self._frame_blocks]

        # Multimodal
        else:
            # Identify patients for which all required video angles are given
            matching_patient_IDs = Helpers.filter_patients_by_video_angle(patient_IDs, video_angles)

            curr_index = 0

            # For each matching patient, save frame blocks for each angle
            for id in matching_patient_IDs:
                self._patients[id] = dict()

                min_blocks = 1000000

                for angle in video_angles:
                    video_name = "{}{}".format(id, angle)

                    blocks = HeartEchoDataset._split_array(all_videos[video_name], frame_block_size, False)

                    if len(blocks) < min_blocks:
                        min_blocks = len(blocks)

                    self._patients[id][angle] = blocks

                    # Do resize
                    if self._resize is not None:
                        if frame_block_size > 1:
                            self._patients[id][angle] = [
                                VideoUtilities.resize_video(blk, self._resize[0], self._resize[1]) for blk in
                                self._patients[id][angle]]
                        else:
                            self._patients[id][angle] = [
                                ImageUtilities.resize_image(blk, self._resize[0], self._resize[1]) for blk in
                                self._patients[id][angle]]

                # Ensure same number of blocks
                for angle in video_angles:
                    self._patients[id][angle] = self._patients[id][angle][:min_blocks]

                # Determine the label
                label_value = self._label_source.get_label_for_patient(id, label_type)

                # Update index list
                for j in range(min_blocks):
                    self._index_list.append((id, j, label_value))

                curr_index += min_blocks

            self._num_samples = len(self._index_list)

        # Balancing
        self._selected_rows = np.arange(self._num_samples)

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

    def get_data(self):
        if not self._multimodal:
            return np.array(np.array(self._frame_blocks)[self._selected_rows]), np.array(self._labels)[
                self._selected_rows]
        else:
            acc_list = []
            ret_list = []
            labels_list = []

            for i in self._selected_rows:
                index_item = self._index_list[i]
                acc_list.append([self._patients[index_item[0]][angle][index_item[1]] for angle in self._video_angles])
                labels_list.append(index_item[2])

            for i in range(len(self._video_angles)):
                ret_list.append([item[i] for item in acc_list])

            return np.array(ret_list), np.array(labels_list)

    def get_label_balance(self):
        if self._label_type in [LABELTYPE.PRETERM_BINARY, LABELTYPE.VISIBLE_FAILURE_WEAK,
                                LABELTYPE.VISIBLE_FAILURE_STRONG]:
            if self._multimodal:
                sum_total = 0

                for idx in self._selected_rows:
                    sum_total += self._index_list[idx][2]

                return sum_total / self._num_samples

            else:
                return np.sum(np.array(self._labels))[self._selected_rows] / self._num_samples
        else:
            raise ValueError("Only supported for PRETERM_BINARY, VISIBLE_FAILURE_WEAK, VISIBLE_FAILURE_STRONG")

    @staticmethod
    def _split_array(array, block_size, keep_remainder=True):
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
