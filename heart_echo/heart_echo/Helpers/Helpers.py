import multiprocessing as mp
import os
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
import time
from itertools import repeat
from tqdm import tqdm
from pathlib import Path

from ..Processing import FrameLoader, VideoUtilities, ImageUtilities, ArcEstimationError


class Helpers:
    ALL_VIDEO_ANGLES = ["LA", "KAKL", "KAPAP", "KAAP", "CV"]
    VIDEO_TYPES = ["mp4", "avi"]
    PROCESSED_FILE = "processed.txt"
    FAILED_FILE = "failed.txt"
    WARNING_FILE = "warning.txt"

    @staticmethod
    def list_from_file(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_all_videos(cache_dir, videos_dir, scale_factor, test, procs=4):
        # Expand directories
        cache_dir = os.path.join(os.path.expanduser(cache_dir), str(scale_factor))
        videos_dir = os.path.expanduser(videos_dir)

        # Load video file list. This list contains all possible videos. However, it is possible some could not be
        # pre-processed, due to insufficient contrast, or other edge cases. So we have to check for these, and then
        # return as many videos as possible
        video_list = Helpers.get_video_list_from_csv(test)
        found_videos = []

        # Check if cache_dir exists and if so, verify that it contains a "processed.txt" file
        if os.path.exists(cache_dir) and Path(os.path.join(cache_dir, Helpers.PROCESSED_FILE)).is_file():
            # Load list of processed videos
            processed_list = Helpers.list_from_file(os.path.join(cache_dir, Helpers.PROCESSED_FILE))

            for video in video_list:
                # Check if video should exist, but doesn't
                if video in processed_list:
                    if not os.path.exists(os.path.join(cache_dir, video) + ".npy"):
                        Helpers.rebuild_video_cache(cache_dir, videos_dir, scale_factor, test, procs)
                        break
                    else:
                        found_videos.append(video)

        else:
            Helpers.rebuild_video_cache(cache_dir, videos_dir, scale_factor, test, procs)

        # Load videos from cache
        ret_dict = dict()

        # Return found videos
        for video in found_videos:
            video_array = np.load(os.path.join(cache_dir, video) + ".npy")
            ret_dict[video] = video_array

        return ret_dict

    @staticmethod
    def get_video_list_from_csv(test):
        videos = []

        path = None
        if test == False: path = "video_angles_list.csv"
        else: path = "video_angles_list_test.csv"

        table = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), path),
                            usecols=["Patient"] + Helpers.ALL_VIDEO_ANGLES, index_col="Patient",
                            na_filter=True)

        for idx, row in table.iterrows():
            for view, val in row.items():
                if not pd.isna(val):
                    videos.append("{}{}".format(idx, view))

        return videos

    @staticmethod
    def get_video_name(filename, videos_dir):
        video_name = None

        for format in Helpers.VIDEO_TYPES:
            video_name_test = "{}.{}".format(filename, format)
            if os.path.exists(os.path.join(videos_dir, video_name_test)):
                video_name = video_name_test
                break

        if video_name is None:
            raise ValueError("Video for {} not found".format(filename))

        return video_name

    @staticmethod
    def _video_segment_loop(videos_dir, video_list, scale_factor):
        videos = []
        segmented_points = []
        failed_videos = []

        for video in video_list:
            try:
                video_with_extension = Helpers.get_video_name(video, videos_dir)
                video_frames, seg_points = Helpers.load_video(os.path.join(videos_dir, video_with_extension),
                                                              scale_factor)
                m, b = VideoUtilities.calculate_line_parameters(*seg_points)
                video_frames = [ImageUtilities.fill_side_of_line(frame, m, b) for frame in video_frames]
                videos.append(video_frames)
                segmented_points.append(seg_points)

            except IndexError as ie:
                print("Video %s failed with index error %s" % (video, ie))
                failed_videos.append(video)

                # Add dummy values
                videos.append([])
                segmented_points.append(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
                continue

            except ArcEstimationError as aee:
                print("Video %s failed to estimate arc: %s" % (video, aee))
                failed_videos.append(video)

                # Add dummy values
                videos.append([])
                segmented_points.append(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
                continue

        return videos, segmented_points, failed_videos

    @staticmethod
    def rebuild_video_cache(cache_dir, videos_dir, scale_factor, test, procs=4):
        def onerror(func, path, exc_info):
            """
            Error handler for ``shutil.rmtree``.

            If the error is due to an access error (read only file)
            it attempts to add write permission and then retries.

            If the error is for another reason it re-raises the error.

            Usage : ``shutil.rmtree(path, onerror=onerror)``
            """
            import stat
            if not os.access(path, os.W_OK):
                # Is the error an access error ?
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, False, onerror=onerror)

        os.makedirs(cache_dir)

        # Load video file list
        video_list = Helpers.get_video_list_from_csv(test)

        # Crop and segment videos
        print("Cropping and segmenting videos...")
        videos = []
        segmented_points = []
        failed_videos = []

        split_video_list = np.array_split(video_list, len(video_list))

        print("Multiprocessing with {} procs".format(procs))

        with mp.Pool(procs) as pool:
            result = pool.starmap_async(Helpers._video_segment_loop,
                                        zip(repeat(videos_dir), split_video_list, repeat(scale_factor)), chunksize=1)

            # TODO: Find a better way to measure progress without checking a private member
            num_chunks = result._number_left
            pbar = tqdm(total=num_chunks)

            mp_done = False
            curr_num_done = 0

            while mp_done is not True:
                time.sleep(1)

                num_done = num_chunks - result._number_left

                # Update progress
                curr_diff = num_done - curr_num_done

                if curr_diff != 0:
                    pbar.update(curr_diff)
                    curr_num_done = num_done

                if num_done == num_chunks:
                    mp_done = True

            pbar.close()

        # Join results

        min_num_frames = 1000  # big number

        for tup in result.get():
            videos += tup[0]
            segmented_points += tup[1]
            failed_videos += tup[2]

        # not all videos have same number of frames (but we need that for np.array contructions further down)
        trimmed_videos = []
        for video in videos:
            trimmed_videos.append(video[:min_num_frames])
        videos = trimmed_videos

        # Determine distribution of top line of cut-off videos
        failed_indices = [video_list.index(f) for f in failed_videos]
        successful_indices = list(range(len(video_list)))

        # Collect videos that might not have been properly processed
        warning_videos = []

        for fi in failed_indices:
            successful_indices.remove(fi)

        # NOTE: Lucas added dtype=object for both array constructions to make error about inhomogeneous dimensions go away. 
        dist_mu, dist_sigma = Helpers._get_cut_off_distribution(np.array(videos, dtype=object)[successful_indices],  # NOTE: edited
                                                                np.array(segmented_points, dtype=object)[successful_indices])  # NOTE: edited

        # Segment videos, cropping the top if necessary
        for i, video_name in enumerate(tqdm(video_list)):
            # Skip video if segmentation failed
            if video_name in failed_videos:
                print("Warning - video {} failed segmentation".format(video_name))
                continue

            segmented_video = VideoUtilities.segment_individual_echo_video(videos[i], dist_mu, dist_sigma,
                                                                           segmented_points[i])

            # Check if the sides have more pixels than expected
            if not VideoUtilities.sanity_check_video(segmented_video):
                print("Warning - video {} has > 1 pixels on side columns".format(video_name))
                warning_videos.append(video_name)

            np.save(os.path.join(cache_dir, video_name), segmented_video)

        # Save processed, failed and warning files
        with open(os.path.join(cache_dir, Helpers.PROCESSED_FILE), "w") as p_f:
            p_f.write("\n".join([video_list[i] for i in successful_indices]))

        with open(os.path.join(cache_dir, Helpers.FAILED_FILE), "w") as f_f:
            f_f.write("\n".join(failed_videos))

        with open(os.path.join(cache_dir, Helpers.WARNING_FILE), "w") as w_f:
            w_f.write("\n".join(warning_videos))

    @staticmethod
    def _get_cut_off_distribution(videos, segmented_points):
        # Get amount of ECHO visible in the first line, if any
        top_line_lengths = [VideoUtilities.get_top_length(videos[i], segmented_points[i]) for i in range(len(videos))]

        # Convert this to a proportion of the height (y3 - y1)
        height_proportions = np.array(
            [top_line_lengths[i] / (segmented_points[i][2][0] - segmented_points[i][0][0]) for i in
             range(len(top_line_lengths))])

        # Discard zeros
        height_proportions = height_proportions[height_proportions.nonzero()[0]]

        # Return distribution parameters
        return np.mean(height_proportions), np.std(height_proportions)

    @staticmethod
    def load_video(video_name, scale_factor=1.0, debug=False):
        cropped_frames = Helpers.load_cropped_video(video_name, scale_factor)
        video_mask = VideoUtilities.get_video_mask(cropped_frames, threshold=30)
        segmented_points = ImageUtilities.find_echo_segmentation_points(video_mask, threshold=30, debug=debug)

        return cropped_frames, segmented_points

    @staticmethod
    def load_cropped_video(video_name, scale_factor=1.0):
        frames = FrameLoader.load_frames(video_name)
        frames = VideoUtilities.convert_to_gray(frames)  # TODO: make this an option
        resized_frames = VideoUtilities.resize_video(frames, int(frames.shape[2] * scale_factor),
                                                     int(frames.shape[1] * scale_factor))
        cropped_frames = ImageUtilities.crop_image(resized_frames, 0, 0.05, 0.01, 0.07)

        return cropped_frames

    @staticmethod
    def filter_patients_by_video_angle(patient_IDs, video_angles, test):

        path = None
        if test == False: path = "video_angles_list.csv"
        else: path = "video_angles_list_test.csv"

        videos = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), path),
                             usecols=["Patient"] + video_angles, index_col="Patient", na_filter=True)

        videos = videos.dropna()

        section = set(patient_IDs).intersection(set(videos.index.values))

        return section

    @staticmethod
    def get_id_and_angle_from_name(name):
        m = re.search('(\d+)(.+)', name)

        return int(m.group(1)), m.group(2)
