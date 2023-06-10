import sys
import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import time
import itertools
import pickle

# import imageio
# import matplotlib.pyplot as plt
import numpy as np
import pims
# import skimage.io
import torch
from torch.utils.data import DataLoader
import tqdm
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
from detectron2_extensions.config import get_cfg as get_detectron_cfg
from detectron2.data.build import trivial_batch_collator
from detectron2.utils import comm
from detectron2.data.samplers import InferenceSampler
from scipy.signal import find_peaks, medfilt
from vq2d.baselines import (
    create_similarity_network,
    convert_annot_to_bbox,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    SiamPredictor,
    SiamPredictorPK,
)
from vq2d.structures import BBox
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack
from vq2d.data.vq2d_inference_dataloader import (
    # VQ2DInferenceDataset,
    VQ2DInferenceDataset2,
    # FrameDataset,
    FrameDataset2,
    process_batch,
)
from vq2d.tracking import Tracker
from einops import asnumpy
setup_logger()

import hydra
from omegaconf import DictConfig, OmegaConf


SKIP_UIDS = {None, }


def setup_d2_cfg(cfg):
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.merge_from_file(cfg.model.config_path)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.score_thresh
    detectron_cfg.MODEL.WEIGHTS = cfg.model.checkpoint_path
    detectron_cfg.MODEL.DEVICE = "cuda"
    detectron_cfg.MODEL.ROI_SIAMESE_HEAD.TOKEN_NUMBER_PER_IMAGE = cfg.model.num_token
    detectron_cfg.TEST.DETECTIONS_PER_IMAGE = cfg.model.num_token
    detectron_cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    detectron_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    # detectron_cfg.INPUT.FORMAT = "RGB"
    return detectron_cfg


def get_annotations(cfg):
    # Load annotations
    annot_path = osp.join(cfg.data.annot_root, f"{cfg.data.split}_annot.json")
    with open(annot_path, "r") as fp:
        annotations = json.load(fp)

    # evaluation for a part of video
    if cfg.data.n_part > 1:
        start = int(len(annotations) * cfg.data.part / cfg.data.n_part)
        end = int(len(annotations) * (cfg.data.part + 1) / cfg.data.n_part)
        if end > len(annotations):
            end = len(annotations)
        annotations = annotations[start:end]
    if cfg.data.debug_mode:
        # import pdb; pdb.set_trace()
        annotations = annotations[3:cfg.data.debug_count]
    elif cfg.data.subsample:
        annotations = annotations[::3]
    print("Total annotations:", len(annotations))

    # Load prior predictions
    if cfg.data.prior_pred_path is not None:
        prior_pred_path = osp.join(
            cfg.data.prior_pred_path,
            f"vq_stats_{cfg.data.split}_{cfg.data.part}.json.gz"
        )
        with gzip.open(prior_pred_path, "r") as fp:
            prior_pred = json.load(fp)
    else:
        prior_pred = None
    # Load dino predictions
    if cfg.dino_rerank:
        dino_pred_path = osp.join(
            cfg.data.prior_pred_path,
            f"vq_stats_{cfg.data.split}_{cfg.data.part}_dino_scores_all.pkl"
        )
        with open(dino_pred_path, "rb") as fp:
            dino_pred = pickle.load(fp)
    else:
        dino_pred = None
    return annotations, prior_pred['predictions'], dino_pred


def peaks_and_post_processing(all_bboxes, all_scores, cfg):
    start_time = time.time()
    # peak_bboxs_all = []
    # peak_scores_all = []
    # # gt_bboxs_all = []
    # # gt_scores_all = []
    # peaks_all = []
    sig_cfg = cfg.signals

    score_signal = []
    for sc in all_scores:
        if len(sc) == 0:
            score_signal.append(0.0)
        else:
            score_signal.append(np.max(sc).item())
    ks = sig_cfg.smoothing_sigma
    score_signal_sm = medfilt(score_signal, kernel_size=ks)

    peaks, _ = find_peaks(
        score_signal_sm,
        height=sig_cfg.height,
        distance=sig_cfg.distance,
        prominence=sig_cfg.prominence,
        width=sig_cfg.width,
    )
    # overide peaks to be all frames
    # peaks = np.arange(len(score_signal))
    peaks = peaks.tolist()
    if len(peaks) > 0:
        peak_bbox = all_bboxes[peaks[-1]][0]
        # peak_frames = [clip_frames[init_state.fno] for init_state in peak_bboxs]
        peak_score = all_scores[peaks[-1]][0]
    else:
        peak_bbox = all_bboxes[-1][0]
        # peak_frames = [clip_frames[peak_bboxs[0].fno]]
        try:
            peak_score = all_scores[-1][0]
        except IndexError:
            peak_score = 1.e-6
        # gt_fnos = [rt["frame_number"] for rt in gt_rt]

        # if len(gt_fnos) > 0:
        #     gt_bboxs = [bbs[gt_fno] for gt_fno in gt_fnos]
        #     gt_scores = [scs[gt_fno] for gt_fno in gt_fnos]
        # else:
        #     gt_bboxs = [bbs[-1]]
        #     gt_scores = [scs[-1]]

        # gt_bboxs_all.append(gt_bboxs)
        # gt_scores_all.append(gt_scores)
    return peak_bbox, peak_score, peaks, time.time() - start_time


def get_trace(all_scores, all_bboxes):
    scores = []
    fnos = []
    for scs, bbs in zip(all_scores, all_bboxes):
        score = []
        fno = []
        for sc, bb in zip(scs, bbs):
            score.append(sc[0])
            fno.append(bb[0].fno.item())
        scores.append(score)
        fnos.append(fno)
    return scores, fnos


def test(predictor, cfg, d2_cfg, tracker, similarity_net):
    # pred_sc = []
    # pred_bb = []
    pred_ps = []
    pred_response_track = []
    visual_crop_boxes = []
    gt_response_track = []
    dataset_uids = []
    n_accessed_frames_per_sample = []
    n_total_frames_per_sample = []
    # gt_pred_sc = []
    # gt_pred_bb = []
    # gt_rt = []
    # vc_boxes = []
    # dataset_uids_all = []
    # pred_tr_scs = []
    # pred_tr_fnos = []
    # Load annotations
    if cfg.signals.smoothing_sigma % 2 == 0:
        cfg.signals.smoothing_sigma += 1
    annotations, prior_pred, dino_pred = get_annotations(cfg)
    if prior_pred is not None:
        prior_pred = [dict(zip(prior_pred, i)) for i in zip(*prior_pred.values())]
    if dino_pred is not None:
        dino_pred = [dino_pred[a['dataset_uid']] for a in annotations]
    # annotations List[Dict] dict_keys(['metadata', 'clip_uid', 'query_set', 'query_frame', 'response_track', 'visual_crop', 'object_title', 'dataset_uid'])
    # prior_pred List[Dict] dict_keys(['predicted_scores', 'predicted_bboxes', 'predicted_peaks', 'groundtruth_response_tracks', 'visual_crop', 'dataset_uids', 'predicted_trace_scores', 'predicted_trace_fnos'])
    # dino_pred List[List[np.ndarray]]] np.ndarray shape (num_token,)
    # import pdb; pdb.set_trace()
    # annotations = annotations[14:]
    # prior_pred = prior_pred[14:]
    # dino_pred = dino_pred[14:]

    print([a["dataset_uid"] for a in annotations[:cfg.num_gpus]])
    root_path = cfg.data.data_root
    dataset = VQ2DInferenceDataset2(d2_cfg, annotations, root_path)
    sampler = InferenceSampler(len(dataset))
    # collate_fn = process_batch
    data_loader = DataLoader(
        dataset,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        batch_size=1,
        collate_fn=trivial_batch_collator,
    )
    # Run inference
    abs_start_time = time.time()
    start_time = time.time()
    for i, data_dicts in enumerate(data_loader):
        data_load_time = time.time() - start_time
        all_bboxes = prior_pred[i]["predicted_bboxes"]
        prior_scores = prior_pred[i]["predicted_scores"]
        if dino_pred is not None:
            dino_scores = dino_pred[i]
        # geometric mean between prior_scores and dino_scores making sure each is at least 1e-6
        prior_scores = [np.clip(np.array(prior_score), 1.e-6, None) for prior_score in prior_scores]
        if dino_pred is not None:
            dino_scores = [np.clip(dino_score, 1.e-6, None) for dino_score in dino_scores]
            all_scores = [np.sqrt(prior_score * dino_score) for prior_score, dino_score in zip(prior_scores, dino_scores)]
        else:
            all_scores = prior_scores
        # all_bboxes, all_scores, xdata_time, run_time = run_test_loader(predictor, cfg, d2_cfg, data_dicts)
        # all_bboxes: List[List[List[BBox]]] 10 detections per frame it seems
        # all_scores: List[List[List[float]]] 10 scores per frame it seems
        # pred_tr_sc, pred_tr_fno = get_trace(all_scores, all_bboxes)
        # import pdb; pdb.set_trace()
        pred_peak_bbox, pred_peak_score, pred_peaks, pp_time = peaks_and_post_processing(
            all_bboxes, all_scores, cfg)
        clip_frames = data_dicts[0]["clip_frames"].get_batch(
            data_dicts[0]["search_window"]).asnumpy()

        if len(pred_peaks) > 0:
            # then run tracker
            init_state = pred_peak_bbox
            init_state = BBox(**init_state)
            init_frame = clip_frames[init_state.fno]
            pred_rt, pred_rt_vis = tracker(
                init_state, init_frame, clip_frames, similarity_net, cfg.machine_rank
            )
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]
            pred_response_track.append(pred_rts)
        else:
            pred_rt = [BBox(**pred_peak_bbox)]
            pred_rt_vis = []
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]
            pred_response_track.append(pred_rts)
        # import pdb; pdb.set_trace()

        visual_crop = data_dicts[0]["visual_crop"]
        visual_crop_boxes.append(convert_annot_to_bbox(visual_crop))
        gt_response_track.append(
            ResponseTrack(
                [convert_annot_to_bbox(rf) for rf in data_dicts[0]["response_track"]]
            )
        )
        dataset_uids.append(data_dicts[0]["dataset_uid"])
        # visual_crops = [data_dict["visual_crop"] for data_dict in data_dicts]

        # pred_sc.extend(pred_peak_scores)
        # pred_bb.extend(pred_peak_bboxes)
        # gt_pred_sc.extend(gt_peak_scores)
        # gt_pred_bb.extend(gt_peak_bboxes)

        pred_ps.extend(pred_peaks)

        accessed_frames = set()
        for bboxes in all_bboxes:
            accessed_frames.add(bboxes[0]['fno'])
        for rt in pred_rts:
            for bbox in rt.bboxes:
                accessed_frames.add(bbox.fno)
        n_accessed_frames = len(accessed_frames)
        n_total_frames = data_dicts[0]['query_frame']
        n_accessed_frames_per_sample.append(n_accessed_frames)
        n_total_frames_per_sample.append(n_total_frames)
        # pred_tr_scs.extend(pred_tr_sc)
        # pred_tr_fnos.extend(pred_tr_fno)
        # gt_rt.extend(gt_response_tracks)
        # vc_boxes.extend(visual_crops)
        # dataset_uids_all.extend(dataset_uids)

        # print a summary of time taken for this step, the current rank and approximate time remaining
        print(
            "Rank: {}, Iter: {}/{}, Time: {:.2f}, Data load time: {:.2f}, PP time: {:.2f}".format(
                comm.get_rank(), i + 1, len(data_loader), time.time() - abs_start_time,
                data_load_time, pp_time
            )
        )
        start_time = time.time()
    return (
        pred_response_track,
        gt_response_track,
        visual_crop_boxes,
        dataset_uids,
        n_accessed_frames_per_sample,
        n_total_frames_per_sample,
    )


def run_test_loader(predictor, cfg, d2_cfg, data_dicts):
    all_bboxes = []
    all_scores = []
    for data_dict in data_dicts:
        ret_bboxes = []
        ret_scores = []
        reference = data_dict["reference"]
        start_time = time.time()
        print('here')
        frames_dset = FrameDataset2(data_dict["clip_frames"], data_dict['search_window'][-1], d2_cfg, data_dict["oheight"], data_dict["owidth"])
        # frames_dset = torch.utils.data.Subset(frames_dset, [1732, 1733])
        frames_dataloader = DataLoader(
            frames_dset,
            num_workers=0,
            batch_size=cfg.data.rcnn_batch_size,
            collate_fn=process_batch,
        )
        time_to_init_extra_data = time.time() - start_time
        start_time = time.time()
        # start_time1 = time.time()
        for frame_chunk, fnos, hs, ws in tqdm.tqdm(frames_dataloader, total=len(frames_dataloader)):
            # print("Time to load frames: {:.2f}".format(time.time() - start_time1))
            # run_time_start = time.time()
            titles = ["title"] * len(frame_chunk)
            # import pdb; pdb.set_trace()
            all_output = predictor(frame_chunk, reference, hs, ws, titles)
            # unpack outputs in normal order using range
            for i in range(len(all_output)):
                out = all_output[i]['instances']
                fno = fnos[i]
                ret_bbs = (asnumpy(out.pred_boxes.tensor).astype(int).tolist())
                ret_bbs = [BBox(fno, *bb) for bb in ret_bbs]
                ret_scs = asnumpy(out.scores).tolist()
                ret_bboxes.append(ret_bbs)
                ret_scores.append(ret_scs)
            # print("Time to run: {:.2f}".format(time.time() - run_time_start))
            # start_time1 = time.time()
        all_bboxes.append(ret_bboxes)
        all_scores.append(ret_scores)
        time_to_run = time.time() - start_time
    return all_bboxes, all_scores, time_to_init_extra_data, time_to_run


def main_worker(cfg: DictConfig) -> None:
    if os.path.exists(cfg.logging.stats_save_path):
        return
    print(comm.get_rank())
    # return
    d2_cfg = setup_d2_cfg(cfg)
    # predictor = SiamPredictorPK(d2_cfg)
    similarity_net = create_similarity_network()
    similarity_net.eval()
    similarity_net.to(cfg.machine_rank)
    tracker = Tracker(cfg)
    predictor = None
    output_local = test(predictor, cfg, d2_cfg, tracker, similarity_net)
    # unpack output
    (
        pred_rt,
        gt_rt,
        vc_boxes,
        duids,
        acc_frames,
        tot_frames,
    ) = output_local
    metrics = compute_visual_query_metrics(pred_rt, gt_rt, vc_boxes, acc_frames, tot_frames)
    predictions = {
        "predicted_response_track": pred_rt,
        "ground_truth_response_track": gt_rt,
        "visual_crop": vc_boxes,
        "dataset_uids": duids,
        "accessed_frames": acc_frames,
        "total_frames": tot_frames,
    }
    # need to collect outputs from all ranks to rank 0
    # comm.synchronize()
    # if comm.is_main_process():
    #     pred_sc = comm.gather(pred_sc_local, dst=0)
    #     pred_bb = comm.gather(pred_bb_local, dst=0)
    #     pred_ps = comm.gather(pred_ps_local, dst=0)
    #     # gt_pred_sc = comm.gather(gt_pred_sc_local, dst=0)
    #     # gt_pred_bb = comm.gather(gt_pred_bb_local, dst=0)
    #     gt_rt = comm.gather(gt_rt_local, dst=0)
    #     vc_boxes = comm.gather(vc_boxes_local, dst=0)
    #     duids = comm.gather(duids_local, dst=0)
    #     pred_tr_scs = comm.gather(pred_tr_scs_local, dst=0)
    #     pred_tr_fnos = comm.gather(pred_tr_fnos_local, dst=0)
    #     # now make a single list of all predictions
    #     pred_sc = list(itertools.chain(*pred_sc))
    #     pred_bb = list(itertools.chain(*pred_bb))
    #     pred_ps = list(itertools.chain(*pred_ps))
    #     # gt_pred_sc = list(itertools.chain(*gt_pred_sc))
    #     # gt_pred_bb = list(itertools.chain(*gt_pred_bb))
    #     gt_rt = list(itertools.chain(*gt_rt))
    #     vc_boxes = list(itertools.chain(*vc_boxes))
    #     duids = list(itertools.chain(*duids))
    #     pred_tr_scs = list(itertools.chain(*pred_tr_scs))
    #     pred_tr_fnos = list(itertools.chain(*pred_tr_fnos))

    #     # import pdb; pdb.set_trace()
    #     predictions = {
    #         "predicted_scores": pred_sc,
    #         "predicted_bboxes": [[[b.to_json() for b in bb] for bb in bbs] for bbs in pred_bb],
    #         "predicted_peaks": pred_ps,
    #         # "groundtruth_predicted_scores": gt_pred_sc,
    #         # "groundtruth_predicted_bboxes": [[[b.to_json() for b in bb] for bb in bbs] for bbs in gt_pred_bb],
    #         "groundtruth_response_tracks": gt_rt,
    #         "visual_crop": vc_boxes,
    #         "dataset_uids": duids,
    #         "predicted_trace_scores": pred_tr_scs,
    #         "predicted_trace_fnos": pred_tr_fnos,
    #     }
    # else:
    #     predictions = None
    # comm.synchronize()
    # import pdb; pdb.set_trace()
    # lets save the predictions as json with gzip compression
    predictions = {
        "predicted_response_track": [
            [
                [rt.to_json() for rt in rts]
                for rts in predictions["predicted_response_track"]
            ]
        ],
        "ground_truth_response_track": [
            rt.to_json() for rt in predictions["ground_truth_response_track"]
        ],
        "visual_crop": [vc.to_json() for vc in predictions["visual_crop"]],
        "dataset_uids": predictions["dataset_uids"],
        "accessed_frames": predictions["accessed_frames"],
        "total_frames": predictions["total_frames"],
    }
    outputs = {"predictions": predictions, "metrics": metrics}
    with gzip.open(cfg.logging.stats_save_path, "wt") as fp:
        json.dump(outputs, fp)
    return
    # if comm.is_main_process():
    #     save_output = {"predictions": predictions, "metrics": {}}
    #     with gzip.open(cfg.logging.stats_save_path, "wt") as fp:
    #         json.dump(save_output, fp)
    # comm.synchronize()
    # return {}, predictions


@hydra.main(config_path="vq2d", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    num_gpus = cfg.num_gpus
    num_machines = cfg.num_machines
    machine_rank = cfg.machine_rank
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    dist_url = f"tcp://127.0.0.1:{port}"
    launch(
        main_worker,
        num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        args=(cfg,),
    )


if __name__ == "__main__":
    TIME = time.time()
    main()
    print("Total time: {:.2f}".format(time.time() - TIME))
