import torch
import numpy as np
from detectron2.data.dataset_mapper import DatasetMapper as DatasetMapperBase
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.config import configurable

from .utils import extract_window_with_context


class DatasetMapper(DatasetMapperBase):
    @configurable
    def __init__(
            self,
            *,
            reference_p: int = 16,
            reference_size: int = 224,
            reference_pad_value: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.reference_p = reference_p
        self.reference_size = reference_size
        self.reference_pad_value = reference_pad_value

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            "reference_p": cfg.INPUT.REFERENCE_CONTEXT_PAD,
            "reference_size": cfg.INPUT.REFERENCE_SIZE,
            "reference_pad_value": 125,
        })
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        if "visual_crop" in dataset_dict:
            # do stuff
            vc = dataset_dict.pop("visual_crop")
            vc_img = utils.read_image(vc["file_name"], format=self.image_format)
            utils.check_image_size(vc, vc_img)
            vc_img = torch.as_tensor(np.ascontiguousarray(vc_img.transpose(2, 0, 1)))
            vc_bbox = BoxMode.convert(vc["bbox"], vc["bbox_mode"], BoxMode.XYXY_ABS)
            reference = extract_window_with_context(
                vc_img,
                vc_bbox,
                p=self.reference_p,
                size=self.reference_size,
                pad_value=self.reference_pad_value,
            )
            dataset_dict["reference"] = reference
        return dataset_dict


if __name__ == "__main__":
    from detectron2_extensions.config import get_cfg
    from detectron2_extensions.data.dataset_mapper import DatasetMapper
    from detectron2.data.build import get_detection_dataset_dicts
    import vq2d.data
    cfg = get_cfg()
    dataset_dicts = get_detection_dataset_dicts("vq2d_val_pos_frame_vc")
    mapper = DatasetMapper(cfg, is_train=False)
    out = mapper(dataset_dicts[0])
    print(out.keys())
    print(out['reference'].size())
