from detectron2.data.datasets import register_coco_instances
from .coco_vq2d import register_coco_vq2d_instances

register_coco_vq2d_instances(
    "vq2d_val_pos_frame_vc", {}, "/users/prannay/vq2d/vq2d_cvpr/data/val_annot_pos_frame_cocofied_vc.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/", new=False
)
register_coco_vq2d_instances(
    "vq2d_val_pos_frame_only_vc", {}, "/users/prannay/vq2d/vq2d_cvpr/data/val_annot_pos_frame_only_cocofied_vc.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/"
)
register_coco_vq2d_instances(
    "vq2d_val_pos_neg_strict_frame_vc", {}, "/users/prannay/vq2d/vq2d_cvpr/data/val_annot_pos_neg_strict_frame_cocofied_vc.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/"
)
register_coco_vq2d_instances(
    "vq2d_val_pos_neg_loose_frame_vc", {}, "/users/prannay/vq2d/vq2d_cvpr/data/val_annot_pos_neg_loose_frame_cocofied_vc.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/"
)
register_coco_instances(
    "vq2d_val_pos_frame", {}, "/users/prannay/vq2d/vq2d_cvpr/data/val_annot_pos_frame_cocofied.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/"
)
register_coco_instances(
    "vq2d_val_pos_frame_interest", {}, "/work/prannay/ego4d/ckpt/vq_logs/val_annot_pos_frame_cocofied_interest.json", "/scratch/shared/beegfs/prannay/ego4d_data/images_val/"
)
