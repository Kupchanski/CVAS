import logging

import argparse
from contextlib import contextmanager
from utils import get_object_coordinates, adjust_coordinates, generate_result_frame, \
    blur_transformed, stabilise_transformation
from segmentation import get_segmentation
import cv2
import numpy as np
import sys
import cProfile
from utils import get_parent_dir_path, get_center
import time

# Enable logging
logging.basicConfig(
    filename=f'{get_parent_dir_path()}/logs.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

class Args:
    def extract(self, args=None):
        parser = argparse.ArgumentParser(description='A Computer Vision Advertisement Switcher')
        parser.add_argument('-i', help='Input file with original video', required=True)
        parser.add_argument('-o', help='File where output should be saved', required=True)
        parser.add_argument('-r', help='Image that will be used as advertisement replacement', required=True)
        parser.add_argument('-s', help='Show intermediate results, 0 = not showing', default=1, type=int)
        parser.add_argument('-segm', help='Choose segmentation model, 0 - Kirill\s Unet, 1 - Artur\'s net', default=0, type=int)
        parser.add_argument('-colab', help='Is it colab usage?', default=0, type=int, choices=[0, 1])
        parser.add_argument("--world-size", metavar="WS", type=int, default=1, help="Number of GPUs")
        parser.add_argument("--rank", metavar="RANK", type=int, default=0, help="GPU id")
        parser.add_argument("--snapshot", metavar="SNAPSHOT_FILE", type=str, help="Snapshot file to load")
        parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"],
                            default="mean",
                            help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
        parser.add_argument("--flip", action="store_true", help="Use horizontal flipping")
        parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                            default="final",
                            help="How the output files are formatted."
                                 " -- palette: color coded predictions"
                                 " -- raw: gray-scale predictions"
                                 " -- prob: gray-scale predictions plus probabilities")

        parser.parse_args(args, namespace=self)
        return self


def get_video_frames(cap):
    @contextmanager
    def context(video_capture):
        try:
            yield None
        finally:
            video_capture.release()
            logger.info("Input video closed")

    with context(cap):
        frame_counter = 0
        while cap.isOpened():
            frame_counter += 1
            ret, frame = cap.read()
            if not ret:
                break

            # uncomment for dev mode
            # if frame_counter % 200 == 0:
            #     logger.info(f"Read {frame_counter} frames")

            yield frame

        logger.info(f"Read {frame_counter} frames")


def save_video(frames, output_path, frame_rate):
    if len(frames) == 0:
        return

    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (int(frame_width), int(frame_height)))

    out.write(frames[0])

    for i in range(1, len(frames)):
        frame = frames[i]
        # final_frame = stabilise_transformation(frame, frames[i - 1])

        final_frame = frame
        out.write(final_frame)
        logger.info(f"Handled {i} frame")


    logger.info(f"Result saved to {output_path}, {len(frames)} frames")
    out.release()


def show_image(args, image):
    if args.colab == 0:
        cv2.imshow('frame', image)
    else:
        from google.colab.patches import cv2_imshow
        cv2_imshow(image)


class GlobalObject:
    def __init__(self, id, frame_index, coordinates):
        pass

        self.id = id
        self.frames = dict()
        self.add_coords(frame_index, coordinates)

    def add_coords(self, frame_index, coordinates):
        self.frames[frame_index] = coordinates

    def cumulative_center(self, frame_index, last_frames=3):
        centers = []

        for i in range(frame_index - last_frames, frame_index):
            coords = self.frames.get(i)

            if coords is not None:
                centers.append(coords.center)

        return get_center(centers)

    def frames_length(self):
        return len(self.frames)

    def get_frame_coords(self, index):
        return self.frames.get(index)

def generate_objects(frames_points):
    thresh = 40
    objects = dict()

    prev_objects = []

    last_id = 0

    for i in range(len(frames_points)):
        frame_data = frames_points[i]

        for object in frame_data:
            matched = False

            for prev_obj in prev_objects:
                prev_center = prev_obj.cumulative_center()

                if prev_center is not None and cv2.norm(object.center - prev_center, cv2.NORM_L2) < thresh:
                    prev_obj.add_new_points(object)
                    matched = True

            if not matched:
                last_id += 1
                new_obj = GlobalObject(last_id, i, object)
                objects[last_id] = new_obj

    return objects

#@profile
def main():
    args = Args().extract(sys.argv[1:])

    input_video_path = args.i
    output_video_path = args.o
    replacement_path = args.r

    # Read replacement image
    replacement = blur_transformed(cv2.imread(replacement_path), with_edge=True)

    result_frames = []
    segments = []

    input_video_cap = cv2.VideoCapture(input_video_path)

    frame_rate = input_video_cap.get(cv2.CAP_PROP_FPS)
    frames_count =  int(input_video_cap. get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = get_video_frames(input_video_cap)

    frames_objects = []

    for index, frame in enumerate(video_frames):
        start = time.time()

        segment_frame = get_segmentation(frame, args)
        object_coordinates = get_object_coordinates(segment_frame)

        frames_objects.append(adjust_coordinates(object_coordinates))
        result_frames.append(frame.copy())

        logger.info(f"Read {index} frame")

    objects = generate_objects(frames_objects)

    final_objects = dict()

    # Filter objects
    for i in objects:
        if objects[i].frames_length() > 3:
            final_objects[i] = objects[i]

    for index in range(frames_count):
        frame = result_frames[index]
        generated_frame = generate_result_frame(frame, index, replacement, final_objects)

        logger.info(f"Handled {counter} frame")
        counter += 1

        if args.s > 0:

            mask_image = cv2.cvtColor(np.array(segment_frame, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
            show_images = np.concatenate((frame, mask_image, generated_frame), axis=1)
            show_image(args, show_images)

            # if cv2.waitKey(20) & 0xFF == 27:
            #     break

        end = time.time()
        logger.info(f"Tme spent on frame {end - start}")

    save_video(result_frames, output_video_path, frame_rate)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    # pr.logger.info_stats()

