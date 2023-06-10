import json
import imageio as iio
import imageio.v3 as iio3
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm.auto import tqdm
import glob
import time
from collections import defaultdict
import sys

MY_CLIPS_ROOT = "/scratch/shared/beegfs/prannay/ego4d_data/vq2d_clips_val/"
ANNS_PATH = os.path.expanduser(sys.argv[1])
FPS = 2
BUFFER_ABS = 1.0
BUFFER_REL = 0.2
NOT_MOD_SINCE = 10  # minutes
CROPS_PATH = "../data/crops/"
GIFS_PATH = os.path.expanduser(sys.argv[2])
NUM_WORKERS = 48
CROP_SHORT_RESIZE = 256
CROP_LONG_MAX = 512
GIF_SHORT_RESIZE = 480
GIF_LONG_MAX = 1000000000000


def extract_crop(ann):
    crop_filename = ann['dataset_uid'] + "_crop.jpg"
    crop_full_filename = os.path.join(CROPS_PATH, crop_filename)

    img_filename = ann['dataset_uid'] + "_img.jpg"
    img_full_filename = os.path.join(CROPS_PATH, img_filename)

    if os.path.isfile(crop_full_filename) and os.path.isfile(img_full_filename):
        return

    clip_uid = ann['clip_uid']
    crop_metadata = ann['visual_crop']
    video_path = os.path.join(MY_CLIPS_ROOT, clip_uid + ".mp4")
    frame_id = crop_metadata['frame_number']
    with iio3.imopen(video_path, "r", plugin="pyav") as img_file:
        img = np.array(img_file.read(index=frame_id))
    x = crop_metadata['x']
    y = crop_metadata['y']
    w = crop_metadata['width']
    h = crop_metadata['height']
    x_, y_, w_, h_ = int(x // 1), int(y // 1), int(w // 1) + 1, int(h // 1) + 1
    x1_, y1_ = max(x_, 0), max(y_, 0)
    x2_, y2_ = min(x_ + w_, img.shape[1]), min(y_ + h_, img.shape[0])
    crop = img[y1_:y2_ + 1, x1_:x2_ + 1].copy()
    img_rect = cv2.rectangle(img, (x_, y_), (x2_, y2_), (255, 255, 0), 10)

    with open(crop_full_filename, "wb") as f:
        iio3.imwrite(f, crop, extension=".jpg")
    with open(img_full_filename, "wb") as f:
        iio3.imwrite(f, img_rect, extension=".jpg")

    # print(crop.shape, img_rect.shape)
    return


def resize_image(img, shortest_side, max_long_side):
    h, w = img.shape[:2]
    if h < w:
        new_h = shortest_side
        new_w = int(w * (shortest_side / h))
    else:
        new_w = shortest_side
        new_h = int(h * (shortest_side / w))
    if new_w > max_long_side:
        new_w = max_long_side
        new_h = int(h * (max_long_side / w))
    elif new_h > max_long_side:
        new_h = max_long_side
        new_w = int(w * (max_long_side / h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_img


def process_clip(ann, use_abs_buffer=True, write_gif=True, use_iio3=True):

    gif_path = os.path.join(GIFS_PATH, ann['dataset_uid'] + ".gif")
    if os.path.isfile(gif_path):
        return
    clip_uid = ann['clip_uid']
    clip_path = os.path.join(MY_CLIPS_ROOT, clip_uid + ".mp4")
    with open(clip_path, "rb") as fp:
        meta = iio3.immeta(fp, extension=".mp4", plugin="pyav")
        num_frames = int(meta['fps'] * meta['duration'])
    min_frame_id = 0
    max_frame_id = num_frames - 1
    ann_max_frame = ann['response_track'][-1]['frame_number']
    ann_min_frame = ann['response_track'][0]['frame_number']
    if use_abs_buffer:
        start_frame = max(min_frame_id, ann_min_frame - BUFFER_ABS * FPS)
        end_frame = min(max_frame_id, max(ann['query_frame'], ann_max_frame + BUFFER_ABS * FPS))
    else:
        start_frame = max(min_frame_id, ann_min_frame - BUFFER_REL * (ann_max_frame - ann_min_frame))
        end_frame = min(max_frame_id, max(ann['query_frame'], ann_max_frame + BUFFER_REL * (ann_max_frame - ann_min_frame)))

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    frame_num = start_frame
    bbox_offset_start = start_frame - ann_min_frame
    bbox_offset_end = ann_max_frame - ann_min_frame

    if not write_gif:
        return

    gif_writer = iio.get_writer(
        gif_path, mode="I", fps=float(FPS)
    )

    crop_path = os.path.join(CROPS_PATH, ann['dataset_uid'] + "_crop.jpg")
    with open(crop_path, "rb") as fp:
        crop = iio3.imread(fp, extension=".jpg")
    crop = resize_image(crop, CROP_SHORT_RESIZE, CROP_LONG_MAX)

    if use_iio3:
        with iio3.imopen(clip_path, "r", plugin="pyav") as img_reader:
            for offset, frame_id in enumerate(range(start_frame, end_frame + 1), start=bbox_offset_start):
                img = img_reader.read(index=frame_id)
                img[:crop.shape[0], :crop.shape[1]] = crop
                if offset >= 0 and offset <= bbox_offset_end:
                    x, y, w, h = (
                        ann['response_track'][offset]['x'],
                        ann['response_track'][offset]['y'],
                        ann['response_track'][offset]['width'],
                        ann['response_track'][offset]['height'],
                    )
                    x, y, w, h = int(x), int(y), int(w) + 1, int(h) + 1
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 10)
                    img_resized = resize_image(img, GIF_SHORT_RESIZE, GIF_LONG_MAX)
                    gif_writer.append_data(img_resized)
                elif frame_id == ann['query_frame']:
                    x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 10)
                    img_resized = resize_image(img, GIF_SHORT_RESIZE, GIF_LONG_MAX)
                    for _ in range(8):
                        gif_writer.append_data(img_resized)
                else:
                    img_resized = resize_image(img, GIF_SHORT_RESIZE, GIF_LONG_MAX)
                    gif_writer.append_data(img_resized)
    else:
        # OpenCV
        vid = cv2.VideoCapture(clip_path)
        for offset, frame_id in enumerate(range(start_frame, end_frame + 1), start=bbox_offset_start):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, img = vid.read()
            assert ret
            if offset >= 0 and offset <= bbox_offset_end:
                x, y, w, h = (
                    ann['response_track'][offset]['x'],
                    ann['response_track'][offset]['y'],
                    ann['response_track'][offset]['width'],
                    ann['response_track'][offset]['height'],
                )
                x, y, w, h = int(x), int(y), int(w) + 1, int(h) + 1
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 10)
            elif frame_id == ann['query_frame']:
                x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 10)

    gif_writer.append_data(np.zeros_like(img_resized))
    gif_writer.close()
    return


def main():
    print(ANNS_PATH)
    with open(ANNS_PATH, "r") as fp:
        annot = json.load(fp)
    print(len(annot))
    clip_uid2annot = defaultdict(list)
    for a in annot:
        clip_uid2annot[a['clip_uid']].append(a)

    all_files = sorted(glob.glob(os.path.join(MY_CLIPS_ROOT, "*.mp4")))
    # only keep files which have not been modified in 10 minutes
    print("Num of clip files: ", len(all_files))
    all_files = [f for f in all_files if os.stat(f).st_mtime < time.time() - (NOT_MOD_SINCE * 0)]
    print("Num of clip files to address: ", len(all_files))

    clip_basenames = list(map(os.path.basename, all_files))
    annot_to_do = [a for clip in clip_basenames for a in clip_uid2annot[clip[:-4]]]
    print("Num of annotations to address: ", len(annot_to_do))

    print("Extracting crops")
    with Pool(NUM_WORKERS) as p:
        list(p.imap(extract_crop, tqdm(annot_to_do, total=len(annot_to_do))))

    print("Make gifs")
    with Pool(NUM_WORKERS) as p:
        list(p.imap_unordered(process_clip, tqdm(annot_to_do, total=len(annot_to_do))))


if __name__ == "__main__":
    main()
