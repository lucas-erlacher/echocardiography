import ffmpeg
import sys
import numpy as np
from matplotlib import pyplot as plt


class FrameLoader:
    @staticmethod
    def load_frames(video_path):
        width, height, num_frames, frame_rate = FrameLoader.get_video_info(video_path)

        out, err = (
            ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(quiet=True)
        )
        video = (
            np
                .frombuffer(out, np.uint8)
                .reshape([-1, height, width, 3])
        )

        return video

    @staticmethod
    def get_video_info(video_path):
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            print(e.stderr, file=sys.stderr)
            return

        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)

        width = int(video_stream['width'])
        height = int(video_stream['height'])
        num_frames = int(video_stream['nb_frames'])
        frame_rate = int(video_stream['r_frame_rate'][0:video_stream['r_frame_rate'].index('/')])

        return width, height, num_frames, frame_rate

    @staticmethod
    def print_video_info(video_path):
        width, height, num_frames, frame_rate = FrameLoader.get_video_info(video_path)

        print('width: {}'.format(width))
        print('height: {}'.format(height))
        print('num_frames: {}'.format(num_frames))
        print('frame_rate: {}'.format(frame_rate))
