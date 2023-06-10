from . import builtin_add

if __name__ == "__main__":
    from detectron2.data import get_detection_dataset_dicts

    dataset_dicts_pos_old = get_detection_dataset_dicts("vq2d_val_pos_frame_vc", filter_empty=False)
    dataset_dicts_pos_new = get_detection_dataset_dicts("vq2d_val_pos_frame_only_vc", filter_empty=False)
    dataset_dicts_pos_neg = get_detection_dataset_dicts("vq2d_val_pos_neg_strict_frame_vc", filter_empty=False)
    import pdb; pdb.set_trace()