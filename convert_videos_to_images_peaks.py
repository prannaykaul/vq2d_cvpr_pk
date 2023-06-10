"""
Script to extract images from a video
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import glob

import imageio
import pims
import tqdm
from vq2d.baselines.utils import get_image_name_from_clip_uid
import gzip

REDO_PATHS = [
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/7380c46e-cb33-4d49-bf73-a7bfe57c3feb/frame_0001355.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/d690cd03-fdbc-4170-b553-82f097dda3a2/frame_0001028.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/d690cd03-fdbc-4170-b553-82f097dda3a2/frame_0001066.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/95d6eae8-8c61-4078-b19b-bcc64eb65b9c/frame_0000824.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/83e54fdc-6336-439c-b8a5-44682989a329/frame_0000705.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/e203fea6-33df-4ecd-9d38-215d2180ab1e/frame_0001128.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/cff0af8b-1b41-4ed3-823a-af64dc8d8dd0/frame_0000353.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/19ad94cc-3029-4d4b-bcbe-8bad52fe0ea5/frame_0000229.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/27a6ec98-3ec3-4d80-92a0-8da2210e9e65/frame_0000246.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/fb2daacd-0cad-48d1-9687-9c6af95204db/frame_0000578.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/fb2daacd-0cad-48d1-9687-9c6af95204db/frame_0000379.png",
    "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val/fb2daacd-0cad-48d1-9687-9c6af95204db/frame_0000038.png",
]


REDO_CLIPS = collections.defaultdict(list)
for path in REDO_PATHS:
    clip_uid = path.split("/")[-2]
    fno = int(path.split("/")[-1].split("_")[-1].split(".")[0]) - 1
    REDO_CLIPS[clip_uid].append(fno)


def read_video_md(path):
    with imageio.get_reader(path, format="mp4") as reader:
        metadata = reader.get_meta_data()
    return metadata


def save_video_frames(vid_path, frames_to_save):
    # import pdb; pdb.set_trace()
    # frames_to_save_dict = collections.defaultdict(list)
    # for fs in frames_to_save:
    #     frames_to_save_dict[fs["video_fno"]].append(fs["save_path"])
    reader = pims.Video(vid_path)
    for fno, path in frames_to_save:
        # if os.path.isfile(path) and os.path.getsize(path) > 0:
        #     continue
        try:
            f = reader[fno]
        except:
            video_md = read_video_md(path)
            max_fno = int(video_md["fps"] * video_md["duration"])
            print(
                f"===> frame {fno} out of range for video {path} (max fno = {max_fno})"
            )
            continue

        # if not os.path.isfile(path) or os.path.getsize(path) == 0:
        if True:
            imageio.imwrite(path, f)
            # else:
            #     fsize = os.path.getsize(path)
            #     print('file size is ', fsize)


def frames_to_select(
    start_frame: int,
    end_frame: int,
    original_fps: int,
    new_fps: int,
):
    # ensure the new fps is divisible by the old
    assert original_fps % new_fps == 0

    # check some obvious things
    assert end_frame >= start_frame

    num_frames = end_frame - start_frame + 1
    skip_number = original_fps // new_fps
    for i in range(0, num_frames, skip_number):
        yield i + start_frame


def extract_clip_frame_nos(video_md, clip_annotation, save_root):
    """
    Extracts frame numbers corresponding to the VQ annotation for a given clip

    Args:
        video_md - a dictionary of video metadata
        clip_annotation - a clip annotation from the VQ task export
        save_root - path to save extracted images
    """
    clip_uid = clip_annotation["clip_uid"]
    clip_fps = int(clip_annotation["clip_fps"])
    # Select frames for clip
    video_fps = int(video_md["fps"])
    vsf = clip_annotation["video_start_frame"]
    vef = clip_annotation["video_end_frame"]
    video_frames_for_clip = list(frames_to_select(vsf, vef, video_fps, clip_fps))
    # Only save images containing response_track and visual_crop
    # annotation = clip_annotation["annotations"][0]
    frames_to_save = []
    for annotation in clip_annotation["annotations"]:
        for qset_id, qset in annotation["query_sets"].items():
            if not qset["is_valid"]:
                continue
            q_fno = qset["query_frame"] # query frame
            vc_fno = qset["visual_crop"]["frame_number"]
            rt_fnos = [rf["frame_number"] for rf in qset["response_track"]]
            # also add negative frames
            rt_fnos = sorted(
                rt_fnos
            )
            rt_dur = rt_fnos[-1] - rt_fnos[0] + 1

            rtn_fnos = [rt_fno+rt_dur for rt_fno in rt_fnos if rt_fno+rt_dur<q_fno]

            all_fnos = [vc_fno] + rt_fnos + rtn_fnos
            for fno in all_fnos:
                path = os.path.join(save_root, get_image_name_from_clip_uid(clip_uid, fno))
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    continue
                frames_to_save.append(
                    {"video_fno": video_frames_for_clip[fno], "save_path": path}
                )
    return frames_to_save


def batchify_video_uids(video_uids, batch_size):
    video_uid_batches = []
    nbatches = len(video_uids) // batch_size
    if batch_size * nbatches < len(video_uids):
        nbatches += 1
    for batch_ix in range(nbatches):
        video_uid_batches.append(
            video_uids[batch_ix * batch_size : (batch_ix + 1) * batch_size]
        )
    return video_uid_batches


def video_to_image_fn(inputs):
    preds, video_data, args = inputs
    clip_uid = video_data["clip_uid"]
    duid = video_data["dataset_uid"]
    if clip_uid is None:
        return None
    # if clip_uid not in REDO_CLIPS.keys():
        # return None
    # print(clip_uid)
    # Extract frames for a specific video_uid
    video_path = os.path.join(args.ego4d_clips_root, clip_uid + ".mp4")
    if not os.path.isfile(video_path):
        print(f"Missing video {video_path}")
        return None

    # Get list of frames to save for annotated clips
    # video_md = read_video_md(video_path)

    frame_nos_to_save = preds['predicted_peaks']
    # frame_nos_to_save = [a for a in frame_nos_to_save if a in REDO_CLIPS[clip_uid]]
    # print(len(frame_nos_to_save), clip_uid)
    # import pdb; pdb.set_trace()
    paths = [
        os.path.join(args.save_root, get_image_name_from_clip_uid(clip_uid, fno))
        for fno in frame_nos_to_save
    ]
    potential_current_paths = [
        os.path.join(
            "/scratch/shared/beegfs/prannay/ego4d_data/peak_frames_val_v2",
            get_image_name_from_clip_uid(clip_uid, fno),)
        for fno in frame_nos_to_save
    ]

    # if all paths exist return
    # if all([os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths]):
    #     return None
    required_frame_nos_to_save = []
    # frame_nos_to_save = list(zip(frame_nos_to_save, paths))
    for fno, path, potential_current_path in zip(frame_nos_to_save, paths, potential_current_paths):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        if os.path.isfile(potential_current_path) and os.path.getsize(potential_current_path) > 0:
            continue
        required_frame_nos_to_save.append((fno, path))

    # for clip_data in video_data["clips"]:
    # Create root directory to save clip

    os.makedirs(os.path.join(args.save_root, clip_uid), exist_ok=True)
    # Get list of frames to save
    # frame_nos_to_save += extract_clip_frame_nos(video_md, clip_data, args.save_root)

    if len(required_frame_nos_to_save) == 0:
        print(f"=========> No valid frames to read for {duid}!")
        return None
    # print(f"=========> Found valid frames to read for {clip_uid}!")
    save_video_frames(video_path, required_frame_nos_to_save)


def preprocess_data(data):
    data = data['predictions']
    # convert generic Dict[List] to List[Dict]
    out_list = []
    keys = sorted(data.keys())
    for idx in range(len(data[keys[0]])):
        save_dict = {k: data[k][idx] for k in keys}
        out_list.append(save_dict)
    return out_list


def main(args):
    # Load annotations
    annotation_export = []
    trace_files = sorted(
        glob.glob(
            args.pred_path),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    for file_path in tqdm.tqdm(trace_files, total=len(trace_files)):
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
            annotation_export += preprocess_data(data)
    annotation_export = sorted(annotation_export, key=lambda x: x["dataset_uids"])
    print("Number of annotations: ", len(annotation_export))

    with open(args.annot_path, "r") as f:
        raw_annotations = json.load(f)

    raw_annot_metadata = {a['dataset_uid']: a for a in raw_annotations}
    print("===> Loaded annotations")

    # video_uids = sorted([a["clip_uid"] for a in annotation_export])
    os.makedirs(args.save_root, exist_ok=True)
    # # if args.video_batch_idx >= 0:
    #     video_uid_batches = batchify_video_uids(video_uids, args.video_batch_size)
    #     video_uids = video_uid_batches[args.video_batch_idx]
    #     print(f"===> Processing video_uids: {video_uids}")
    # # Get annotations corresponding to video_uids
    # test case
    if args.num_shards > 0:
        annotation_export = annotation_export[args.shard_idx::args.num_shards]
    print("Number of annotations after sharding: ", len(annotation_export))
    inputs = [(video_data, raw_annot_metadata[video_data['dataset_uids']], args) for video_data in annotation_export]
    # _ = video_to_image_fn(inputs[0])
    pool = mp.Pool(args.num_workers)
    _ = list(
        tqdm.tqdm(
            pool.imap_unordered(video_to_image_fn, inputs),
            total=len(inputs),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-batch-idx", type=int, default=-1)
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--annot-path", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--ego4d-clips-root", type=str, required=True)
    parser.add_argument("--video-batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    main(args)
