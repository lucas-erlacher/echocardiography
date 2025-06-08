import numpy as np
import cv2
from functools import reduce
from ..Processing import ImageUtilities


class VideoUtilities:
    @staticmethod
    def get_video_mask(video_frames, threshold=20):
        video_mask = np.zeros_like(video_frames[0])

        for frame in video_frames:
            video_mask = ImageUtilities.get_frame_mask(frame, video_mask, threshold)

        return video_mask

    @staticmethod
    def detect_edges_in_video(video_frames, canny_min, canny_max, blur):
        return [ImageUtilities.detect_edges(frame, canny_min, canny_max, blur) for frame in video_frames]

    @staticmethod
    def find_echo_video_segmentation_points(video_frames, debug=False):
        # Convert video to greyscale if necessary
        if len(video_frames.shape) == 4:
            video_frames = VideoUtilities.convert_to_gray(video_frames)

        # Get the top left, bottom and top right points in the video
        top_left, bottom, top_right = VideoUtilities._get_static_points(video_frames)

        # Get the cone arc parameters
        center, radius, _, _ = VideoUtilities._get_arc(video_frames, top_left, bottom, top_right)

        # Get the cone side line endpoints
        bottom_left, bottom_right, point_count = VideoUtilities._get_line_endpoints(video_frames, top_left, top_right)

        if debug:
            ImageUtilities.show_image(
                ImageUtilities.draw_segmentation_points(video_frames[0], top_left, bottom_left, bottom, bottom_right,
                                                        (0, 0), top_right, center, radius))

        # Get cone endpoints
        bottom_left = ImageUtilities.get_circle_line_intersection(center, radius, top_left, bottom_left)
        bottom_right = ImageUtilities.get_circle_line_intersection(center, radius, top_right, bottom_right)

        # Calculate the 5th point 20% along the right line -- this is used to crop out any text
        point_5 = (
            int(np.round(top_right[0] + 0.2 * (bottom_right[0] - top_right[0]))),
            int(np.round(top_right[1] + 0.2 * (bottom_right[1] - top_right[1]))))

        if debug:
            ImageUtilities.show_image(
                ImageUtilities.draw_segmentation_points(video_frames[0], top_left, bottom_left, bottom, bottom_right,
                                                        point_5, top_right, center, radius))

        return top_left, bottom_left, bottom, bottom_right, point_5, top_right

    @staticmethod
    def _get_static_points(video_frames):
        static_points = [ImageUtilities.get_static_points(frame) for frame in video_frames]
        top_left = reduce(lambda a, b: a if a[1] < b[1] else b, [x[0] for x in static_points])
        bottom = reduce(lambda a, b: a if a[0] > b[0] else b, [x[1] for x in static_points])
        top_right = reduce(lambda a, b: a if a[1] > b[1] else b, [x[2] for x in static_points])

        return top_left, bottom, top_right

    @staticmethod
    def _get_arc(video_frames, top_left, bottom, top_right):
        arcs = []
        for i, frame in enumerate(video_frames):
            try:
                arcs.append(ImageUtilities.get_arc(frame, top_left, bottom, top_right))

            except ValueError:
                continue

        idx = np.argmax([x[3] for x in arcs])

        return arcs[idx]

    @staticmethod
    def _get_line_endpoints(video_frames, top_left, top_right):
        endpoints = []

        for frame in video_frames:
            try:
                endpoints.append(ImageUtilities.get_line_endpoints(frame, top_left, top_right))

            except ValueError:
                continue

        idx = np.argmax([x[2] for x in endpoints])

        return endpoints[idx]

    @staticmethod
    def sanity_check_video(video_frames):
        for frame in video_frames:
            if not ImageUtilities.check_frame(frame):
                return False

        return True

    @staticmethod
    def segment_echo_video(video_frames, point_1, point_2, point_3, point_4, point_5, point_6):
        segmented_frames = [ImageUtilities.crop_image(frame, 0, point_4[1], point_3[0], point_2[1], mode="exact") for
                            frame in video_frames]

        return np.array(segmented_frames)

    @staticmethod
    def segment_individual_echo_video(video_frames, dist_mu, dist_sigma, points):
        # Determine if top should be cropped
        if ImageUtilities.is_frame_top_empty(video_frames[0]):
            sampled_ratio = np.random.normal(dist_mu, dist_sigma)
            top_y = np.round((points[2][0] - points[0][0]) * sampled_ratio).astype(int) + points[0][0]

        else:
            top_y = 0

        segmented_frames = [
            ImageUtilities.crop_image(frame, top_y, points[3][1], points[2][0], points[1][1], mode="exact") for frame in
            video_frames]

        return np.array(segmented_frames)

    @staticmethod
    def calculate_line_parameters(point_1, point_2, point_3, point_4, point_5, point_6):
        m = (point_6[0] - point_5[0]) / (point_6[1] - point_5[1])
        b = int(point_5[1] - (point_5[0] / m))

        return m, b

    @staticmethod
    def play_video(video_frames):
        for frame in video_frames:
            cv2.imshow('video', frame)
            cv2.waitKey(40)

        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image

    @staticmethod
    def convert_to_gray(video_frames):
        return np.array([ImageUtilities.convert_to_gray(frame) for frame in video_frames])

    @staticmethod
    def resize_video(video_frames, width, height, interpolation=cv2.INTER_LINEAR):
        return np.array([ImageUtilities.resize_image(frame, width, height, interpolation) for frame in video_frames])

    @staticmethod
    def draw_echo_segmentation(img_array, point_1, point_2, point_3, point_4, point_5, point_6):
        video = [ImageUtilities.draw_echo_segmentation(frame, point_1, point_2, point_3, point_4, point_5, point_6) for
                 frame in img_array]

        return video

    @staticmethod
    def get_top_length(video_frames, segmented_points):
        # We only need to look at the first frame here. If there are any non-zero points here, then it
        # is a cutoff video. In this case, the top length is then x6 - x1
        if ImageUtilities.is_frame_top_empty(video_frames[0]):
            return 0

        else:
            return segmented_points[5][1] - segmented_points[0][1]
