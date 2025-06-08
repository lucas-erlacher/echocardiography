import argparse
import os.path

import numpy as np
from tqdm import tqdm
from ..Processing import FrameLoader, VideoUtilities, ImageUtilities, ArcEstimationError
from ..Helpers import Helpers, LABELTYPE
from torchvision import transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, help="FQ path to video", required=False)
    parser.add_argument("--views", default=None, type=str, nargs="+", choices=["LA", "KAKL", "KAPAP", "KAAP", "CV"],
                        help="Views to load")
    parser.add_argument("--label", default="ID", type=str,
                        choices=["age", "maturity", "preterm_binary", "preterm_categorical", "failure_weak",
                                 "failure_strong", "view", "ID", "HF", "PH"], help="Label type")
    parser.add_argument("--balance", action="store_true", default=False, help="Balance dataset")
    parser.add_argument("--check_cache", action="store_true", default=False, help="View cache")
    parser.add_argument("--view_warning_videos", action="store_true", default=False, help="View videos with warnings")
    parser.add_argument("--procs", type=int, help="Number of procs", default=4, required=False)
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scaling factor")
    parser.add_argument("--cache_dir", type=str, default="", help="Custom cache directory")

    args = parser.parse_args()
    return args


def pytorch_example(views, label, balance, procs, scale_factor):
    from ..pytorch import HeartEchoDataset
    from torch.utils import data

    # pytorch dataset test example
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}

    # Note that the transform is *necessary*, as the preprocessed frames have only been scaled, not yet resized to all
    # be consistent. The way that this is done is up to the user, and must be specified using a torchvision transform
    # like the below
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(128, 128),
                                                      interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])

    # Example of a PyTorch dataset using a frame block size of 50 (so data points will be videos)
    train_dataset = HeartEchoDataset([30, 31, 33, 200, 201], views, scale_factor=scale_factor, frame_block_size=50,
                                     label_type=label, balance=balance, procs=procs, transform=transform)
    training_generator = data.DataLoader(train_dataset, **params)

    # Example of a PyTorch dataset using a frame block size of 1 (so data points will be images)
    test_dataset = HeartEchoDataset([38, 39, 40, 41, 42], views, scale_factor=scale_factor, frame_block_size=1,
                                    label_type=label, balance=balance, procs=procs, transform=transform)
    test_generator = data.DataLoader(test_dataset, **params)

    # Do training
    for i in range(10):
        for x, y in training_generator:
            # Do stuff...
            # Note: if more than one view is specified, then x[0] is modality 1, x[1] modality 2, etc...

            # Uncomment the following lines below to view the video output of the training loader
            # if train_dataset.is_multimodal():
            #     VideoUtilities.play_video(x[0][0].cpu().detach().numpy())
            # else:
            #     VideoUtilities.play_video(x[0].cpu().detach().numpy())

            pass

    # Do evaluation
    for x, y in test_generator:
        # Test
        # Uncomment to view the image output of the test loader
        # if test_dataset.is_multimodal():
        #     ImageUtilities.show_image(x[0][0].cpu().detach().numpy())
        # else:
        #     ImageUtilities.show_image(x[0].cpu().detach().numpy())
        pass


def numpy_array_example(views, label, balance, procs, scale_factor):
    from ..numpy import HeartEchoDataset

    # Example of a Numpy dataset using a frame block size of 50 (so data points will be videos)
    train_dataset = HeartEchoDataset(
        [30, 31, 33, 34, 35, 36, 37], views, scale_factor=scale_factor, frame_block_size=50, label_type=label,
        balance=balance, procs=procs, resize=(64, 64))

    # Example of a Numpy dataset using a frame block size of 1 (so data points will be images)
    test_dataset = HeartEchoDataset([38, 39, 40, 41, 42], views, scale_factor=scale_factor, frame_block_size=1,
                                    label_type=label, balance=balance, procs=procs, resize=(64, 64))

    # Do training
    train_data, train_labels = train_dataset.get_data()

    for i in range(10):
        # Make batches from train_data, train_labels and feed to model
        # Note: if more than one view is specified, train_data[0] is modality 1, train_data[1] is modality 2, etc...

        # Uncomment the following lines below to view the output of the data loaders
        # if train_dataset.is_multimodal():
        #     VideoUtilities.play_video(train_data[0][0])
        # else:
        #     VideoUtilities.play_video(train_data[0])

        pass

    # Do evaluation
    test_data, test_labels = test_dataset.get_data()

    # Feed to model

    # Uncomment the following lines below to view the output of the data loaders
    # if test_dataset.is_multimodal():
    #     ImageUtilities.show_image(test_data[0][0])
    # else:
    #     ImageUtilities.show_image(test_data[0])
    pass


def view_videos(procs, scale_factor, cache_dir="", file_list=Helpers.PROCESSED_FILE):
    if cache_dir == "":
        cache_dir = "~/.heart_echo"

    cached_videos = Helpers.load_all_videos(cache_dir=cache_dir,
                                            videos_dir="/cluster/work/vogtlab/Projects/Heart_Echo",
                                            scale_factor=scale_factor, procs=procs)

    video_list = Helpers.list_from_file(
        os.path.join(os.path.expanduser(cache_dir), str(scale_factor), file_list))

    for video in video_list:
        # For videos in the warning list, count pixels on each end line and print the counts
        if file_list == Helpers.WARNING_FILE:
            video_frames = cached_videos[video]
            l = 0
            r = 0

            for frame in video_frames:
                left_side = len((frame.T[0] > 20).nonzero()[0])
                l = left_side if left_side > l else l
                right_side = len((frame.T[-1] > 20).nonzero()[0])
                r = right_side if right_side > r else r

            print("Video %s: left: %i right: %i" % (video, l, r))

        # Otherwise just print the video name
        else:
            print("Video %s" % video)

        VideoUtilities.play_video(cached_videos[video])


def view_cached_videos(procs, scale_factor, cache_dir=""):
    view_videos(procs, scale_factor, cache_dir, Helpers.PROCESSED_FILE)


def view_warning_videos(procs, scale_factor, cache_dir=""):
    view_videos(procs, scale_factor, cache_dir, Helpers.WARNING_FILE)


def main():
    # Parse arguments
    args = parse_args()

    if args.check_cache:
        view_cached_videos(args.procs, args.scale_factor, args.cache_dir)
        exit(0)

    # Label
    if args.label == "age":
        label = LABELTYPE.AGE
    elif args.label == "maturity":
        label = LABELTYPE.MATURITY_AT_BIRTH
    elif args.label == "preterm_binary":
        label = LABELTYPE.PRETERM_BINARY
    elif args.label == "preterm_categorical":
        label = LABELTYPE.PRETERM_CATEGORICAL
    elif args.label == "failure_weak":

        label = LABELTYPE.VISIBLE_FAILURE_WEAK
    elif args.label == "failure_strong":
        label = LABELTYPE.VISIBLE_FAILURE_STRONG
    elif args.label == "view":
        label = LABELTYPE.VIEW
    elif args.label == "ID":
        label = LABELTYPE.ID
    elif args.label == "HF":
        label = LABELTYPE.HEART_FAILURE
    elif args.label == "PH":
        label = LABELTYPE.PULMINORY_HYPERTENSION
    else:
        label = LABELTYPE.NONE

    if args.video is not None:
        cropped_frames = Helpers.load_cropped_video(args.video, args.scale_factor)
        video_mask = VideoUtilities.get_video_mask(cropped_frames, threshold=30)
        ImageUtilities.show_image(video_mask)
        # video_mask = ImageUtilities.smooth_image(video_mask)
        # video_mask = ImageUtilities.get_frame_mask(video_mask, video_mask, threshold=1)
        # ImageUtilities.show_image(video_mask)
        segmented_points = ImageUtilities.find_echo_segmentation_points(video_mask, threshold=30, debug=True)

        highlight_video = VideoUtilities.draw_echo_segmentation(cropped_frames, *segmented_points)
        VideoUtilities.play_video(highlight_video)
        m, b = VideoUtilities.calculate_line_parameters(*segmented_points)
        cropped_frames = [ImageUtilities.fill_side_of_line(frame, m, b) for frame in cropped_frames]
        segmented_video = VideoUtilities.segment_echo_video(cropped_frames, *segmented_points)

        if not VideoUtilities.sanity_check_video(segmented_video):
            print("Warning - video {} has > 1 pixels on side columns".format(args.video))

        VideoUtilities.play_video(segmented_video)

    elif args.view_warning_videos:
        view_warning_videos(args.procs, args.scale_factor)

    else:
        pytorch_example(args.views, label, args.balance, args.procs, args.scale_factor)
        numpy_array_example(args.views, label, args.balance, args.procs, args.scale_factor)


if __name__ == '__main__':
    main()
