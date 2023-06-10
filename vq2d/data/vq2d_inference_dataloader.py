import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
# import pims
from ..baselines.utils import get_clip_name_from_clip_uid, extract_window_with_context
from einops import rearrange, asnumpy
from detectron2.config import CfgNode
from typing import List, Dict, Sequence
import detectron2.data.transforms as T
from decord import VideoReader
from detectron2.utils.serialize import PicklableWrapper
import decord
decord.bridge.set_bridge("native")

# class VQ2DInferenceDataset(Dataset):
#     """
#     This dataset will be initialised with an annotation file and a path to
#     videos. It will then load the video and extract the frames as needed
#     depending on the frame of the visual query. Moreover it will apply the
#     required augmentations to the visual crop.
#     Args:
#         cfg (CfgNode): config
#         annotation_file List[Dict]: list of annotations
#         root_path (str): path to the folder containing the videos
#     """
#     def __init__(
#             self,
#             cfg: CfgNode,
#             annotations: List[Dict],
#             root_path: str,
#             recency_factor: float = 1.0,
#             subsampling_factor: float = 1.0
#     ):
#         self.cfg = cfg
#         self.annotations = annotations
#         self.root_path = root_path
#         self.recency_factor = recency_factor
#         self.subsampling_factor = subsampling_factor

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         annotation = self.annotations[idx]
#         clip_uid = annotation["clip_uid"]
#         clip_name = get_clip_name_from_clip_uid(clip_uid)
#         clip_path = os.path.join(self.root_path, clip_name)
#         video_reader = pims.Video(clip_path)
#         query_frame = annotation["query_frame"]
#         visual_crop = annotation["visual_crop"]

#         vcfno = visual_crop['frame_number']
#         clip_frames = video_reader[:max(query_frame, vcfno) + 1]

#         # get the visual crop
#         vcfno = visual_crop['frame_number']
#         owidth, oheight = visual_crop['original_width'], visual_crop['original_height']
#         reference_frame = clip_frames[vcfno]
#         if (reference_frame.shape[0] != oheight) or (reference_frame.shape[1] != owidth):
#             reference_frame = cv2.resize(reference_frame, (owidth, oheight))
#         reference = torch.as_tensor(rearrange(reference_frame, "h w c -> () c h w"))
#         reference = reference.float()
#         ref_bbox = (
#             visual_crop["x"],
#             visual_crop["y"],
#             visual_crop["x"] + visual_crop["width"],
#             visual_crop["y"] + visual_crop["height"],
#         )
#         reference = extract_window_with_context(
#             reference,
#             ref_bbox,
#             self.cfg.INPUT.REFERENCE_CONTEXT_PAD,
#             size=self.cfg.INPUT.REFERENCE_SIZE,
#             pad_value=125,
#         )
#         reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
#         # Define search window
#         search_window = list(range(0, query_frame))
#         # Pick recent k% of frames
#         window_size = int(round(len(search_window) * self.recency_factor))
#         if len(search_window[-window_size:]) > 0:
#             search_window = search_window[-window_size:]
#         # Subsample only k% of frames
#         window_size = len(search_window)
#         idxs_to_sample = np.linspace(
#             0, window_size - 1, int(self.subsampling_factor * window_size)
#         ).astype(int)
#         if len(idxs_to_sample) > 0:
#             search_window = [search_window[i] for i in idxs_to_sample]
#         # print(annotation['dataset_uid'], len(search_window))
#         # reverse colour channels if needed
#         if self.cfg.INPUT.FORMAT == "RGB":
#             reference = reference[:, :, ::-1]
#         reference = torch.as_tensor(
#             reference.astype("float32").transpose(2, 0, 1)
#         )
#         dataset_uid = annotation["dataset_uid"]
#         ret_dict = {
#             "clip_frames": clip_frames[:query_frame],
#             "search_window": search_window,
#             "reference": reference,
#             "query_frame": query_frame,
#             "clip_uid": clip_uid,
#             "oheight": oheight,
#             "owidth": owidth,
#             "dataset_uid": dataset_uid,
#             "response_track": annotation["response_track"],
#             "visual_crop": visual_crop,
#         }
#         return ret_dict


class VQ2DInferenceDataset2(Dataset):
    """
    This dataset will be initialised with an annotation file and a path to
    videos. It will then load the video and extract the frames as needed
    depending on the frame of the visual query. Moreover it will apply the
    required augmentations to the visual crop.
    Args:
        cfg (CfgNode): config
        annotation_file List[Dict]: list of annotations
        root_path (str): path to the folder containing the videos
    """
    def __init__(
            self,
            cfg: CfgNode,
            annotations: List[Dict],
            root_path: str,
            recency_factor: float = 1.0,
            subsampling_factor: float = 1.0
    ):
        self.cfg = cfg
        self.annotations = annotations
        self.root_path = root_path
        self.recency_factor = recency_factor
        self.subsampling_factor = subsampling_factor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # print("here")
        annotation = self.annotations[idx]
        clip_uid = annotation["clip_uid"]
        clip_name = get_clip_name_from_clip_uid(clip_uid)
        clip_path = os.path.join(self.root_path, clip_name)
        clip_frames = VideoReader(clip_path)
        query_frame = annotation["query_frame"]
        visual_crop = annotation["visual_crop"]

        vcfno = visual_crop['frame_number']

        # get the visual crop
        vcfno = visual_crop['frame_number']
        owidth, oheight = visual_crop['original_width'], visual_crop['original_height']
        reference_frame = clip_frames[vcfno].asnumpy()
        if (reference_frame.shape[0] != oheight) or (reference_frame.shape[1] != owidth):
            reference_frame = cv2.resize(reference_frame, (owidth, oheight))
        reference = torch.as_tensor(rearrange(reference_frame, "h w c -> () c h w"))
        reference = reference.float()
        ref_bbox = (
            visual_crop["x"],
            visual_crop["y"],
            visual_crop["x"] + visual_crop["width"],
            visual_crop["y"] + visual_crop["height"],
        )
        # print("here1")
        reference = extract_window_with_context(
            reference,
            ref_bbox,
            self.cfg.INPUT.REFERENCE_CONTEXT_PAD,
            size=self.cfg.INPUT.REFERENCE_SIZE,
            pad_value=125,
        )
        reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
        # Define search window
        search_window = list(range(0, query_frame))
        # Pick recent k% of frames
        window_size = int(round(len(search_window) * self.recency_factor))
        if len(search_window[-window_size:]) > 0:
            search_window = search_window[-window_size:]
        # Subsample only k% of frames
        window_size = len(search_window)
        idxs_to_sample = np.linspace(
            0, window_size - 1, int(self.subsampling_factor * window_size)
        ).astype(int)
        if len(idxs_to_sample) > 0:
            search_window = [search_window[i] for i in idxs_to_sample]
        # print(annotation['dataset_uid'], len(search_window))
        # reverse colour channels if needed
        if self.cfg.INPUT.FORMAT == "BGR":
            reference = reference[:, :, ::-1]
        reference = torch.as_tensor(
            reference.astype("float32").transpose(2, 0, 1)
        )
        dataset_uid = annotation["dataset_uid"]
        # print("here2")
        ret_dict = {
            "clip_frames": clip_frames,
            "search_window": search_window,
            "reference": reference,
            "query_frame": query_frame,
            "clip_uid": clip_uid,
            "oheight": oheight,
            "owidth": owidth,
            "dataset_uid": dataset_uid,
            "response_track": annotation["response_track"],
            "visual_crop": visual_crop,
        }
        return ret_dict


class VQ2DInferenceDataset3(Dataset):
    """
    This dataset will be initialised with an annotation file and a path to
    videos. It will then load the video and extract the frames as needed
    depending on the frame of the visual query. Moreover it will apply the
    required augmentations to the visual crop.
    Args:
        cfg (CfgNode): config
        annotation_file List[Dict]: list of annotations
        root_path (str): path to the folder containing the videos
    """
    def __init__(
            self,
            # cfg: CfgNode,
            annotations: List[Dict],
            preds: List[Dict],
            root_path: str,
            recency_factor: float = 1.0,
            subsampling_factor: float = 1.0
    ):
        # self.cfg = cfg
        self.annotations = annotations
        self.preds = preds
        self.root_path = root_path
        self.recency_factor = recency_factor
        self.subsampling_factor = subsampling_factor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # print("here")
        annotation = self.annotations[idx]
        pred = self.preds[idx]
        clip_uid = annotation["clip_uid"]
        clip_name = get_clip_name_from_clip_uid(clip_uid)
        clip_path = os.path.join(self.root_path, clip_name)
        clip_frames = VideoReader(clip_path)
        query_frame = annotation["query_frame"]
        visual_crop = annotation["visual_crop"]

        vcfno = visual_crop['frame_number']

        # get the visual crop
        vcfno = visual_crop['frame_number']
        owidth, oheight = visual_crop['original_width'], visual_crop['original_height']
        reference_frame = clip_frames[vcfno].asnumpy()
        if (reference_frame.shape[0] != oheight) or (reference_frame.shape[1] != owidth):
            reference_frame = cv2.resize(reference_frame, (owidth, oheight))
        reference = torch.as_tensor(rearrange(reference_frame, "h w c -> () c h w"))
        reference = reference.float()
        ref_bbox = (
            visual_crop["x"],
            visual_crop["y"],
            visual_crop["x"] + visual_crop["width"],
            visual_crop["y"] + visual_crop["height"],
        )
        # print("here1")
        reference = extract_window_with_context(
            reference,
            ref_bbox,
            p=0,
            # self.cfg.INPUT.REFERENCE_CONTEXT_PAD,
            size=224,
            pad_value=0,
        )
        reference = rearrange(asnumpy(reference), "() c h w -> h w c")
        # Define search window
        search_window = list(range(0, query_frame))
        # Pick recent k% of frames
        window_size = int(round(len(search_window) * self.recency_factor))
        if len(search_window[-window_size:]) > 0:
            search_window = search_window[-window_size:]
        # Subsample only k% of frames
        window_size = len(search_window)
        idxs_to_sample = np.linspace(
            0, window_size - 1, int(self.subsampling_factor * window_size)
        ).astype(int)
        if len(idxs_to_sample) > 0:
            search_window = [search_window[i] for i in idxs_to_sample]
        # print(annotation['dataset_uid'], len(search_window))
        # reverse colour channels if needed
        # if self.cfg.INPUT.FORMAT == "BGR":
        #     reference = reference[:, :, ::-1]
        reference = torch.as_tensor(
            reference.astype("float32").transpose(2, 0, 1)
        )
        dataset_uid = annotation["dataset_uid"]
        # print("here2")
        length = min(len(clip_frames), len(search_window))
        ret_dict = {
            "clip_frames": clip_frames.get_batch(range(length)).asnumpy(),
            # "clip_frames": clip_frames,
            "search_window": search_window,
            "reference": reference,
            "query_frame": query_frame,
            "clip_uid": clip_uid,
            "oheight": oheight,
            "owidth": owidth,
            "dataset_uid": dataset_uid,
            "response_track": annotation["response_track"],
            "visual_crop": visual_crop,
            "pred": pred,
        }
        return ret_dict


def process_batch(batch):
    frames = torch.stack([x[0] for x in batch])
    fno = torch.tensor([x[1] for x in batch], dtype=torch.long)
    heights = torch.tensor([x[2] for x in batch], dtype=torch.long)
    widths = torch.tensor([x[3] for x in batch], dtype=torch.long)
    return frames, fno, heights, widths


# class FrameDataset(Dataset):
#     def __init__(
#             self,
#             frames: Sequence[np.ndarray],
#             cfg: CfgNode,
#             oheight: int,
#             owidth: int,
#             # downscale_height: int,
#     ):
#         self.frames = frames
#         self.cfg = cfg
#         self.oheight = oheight
#         self.owidth = owidth
#         # self.downscale_height = downscale_height

#         self.aug = T.ResizeShortestEdge(
#             [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
#         )

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, idx):
#         image = self.frames[idx]
#         fno = image.frame_no
#         if image.shape[:2] != (self.oheight, self.owidth):
#             image = cv2.resize(image, (self.owidth, self.oheight))
#         # reverse colour channels
#         if self.cfg.INPUT.FORMAT == "RGB":
#             image = image[:, :, ::-1]
#         height, width = image.shape[:2]
#         image = self.aug.get_transform(image).apply_image(image)
#         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#         # # Scale-down image to reduce memory consumption
#         # image_scale = float(self.downscale_height) / image.shape[0]
#         # image = cv2.resize(image, None, fx=image_scale, fy=image_scale)
#         # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#         return image, fno, height, width


class FrameDataset2(Dataset):
    def __init__(
            self,
            frames,
            fin_frame: int,
            cfg: CfgNode,
            oheight: int,
            owidth: int,
            # downscale_height: int,
    ):
        self.frames = frames
        self.fin_frame = fin_frame + 1
        self.cfg = cfg
        self.oheight = oheight
        self.owidth = owidth
        # self.downscale_height = downscale_height

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __len__(self):
        return min(self.fin_frame, len(self.frames))

    def __getitem__(self, idx):
        image = self.frames[idx].asnumpy()
        # import pdb; pdb.set_trace()
        # image = image.asnumpy()
        fno = idx
        if image.shape[:2] != (self.oheight, self.owidth):
            image = cv2.resize(image, (self.owidth, self.oheight))
        # reverse colour channels
        if self.cfg.INPUT.FORMAT == "BGR":
            image = image[:, :, ::-1]
        height, width = image.shape[:2]
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("uint8").transpose(2, 0, 1))
        # # Scale-down image to reduce memory consumption
        # image_scale = float(self.downscale_height) / image.shape[0]
        # image = cv2.resize(image, None, fx=image_scale, fy=image_scale)
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return image, fno, height, width
