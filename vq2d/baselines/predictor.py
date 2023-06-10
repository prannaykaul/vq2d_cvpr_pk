from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from detectron2.engine import DefaultPredictor


# Rewrite the SiamPredictor class so we do not have to do all this weird rescaling and
# refactoring
class SiamPredictorPK(DefaultPredictor):
    def __call__(
        self,
        original_images: torch.Tensor,
        visual_crop: torch.Tensor,
        heights: torch.Tensor,
        widths: torch.Tensor,
        titles: Sequence[str] = None,
        return_top_feature: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            original_images (np.ndarray): a list of images of shape (H, W, C).
            visual_crops (np.ndarray): visual crop for the images

        Returns:
            predictions (list[dict]):
                the output of the model for a list of images.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            # we are not going to do all the preproceessing here - done in the dataloader
            inputs = []
            for idx, orig_image in enumerate(original_images):
                # colour channel stuff should be done in the dataloader
                # if self.input_format == "RGB":
                #     orig_image = orig_image[:, :, ::-1]
                #     vis_crop = vis_crop[:, :, ::-1]
                # height, width = orig_image.shape[:2]
                # image = torch.as_tensor(orig_image.astype("float32").transpose(2, 0, 1))
                # reference = torch.as_tensor(vis_crop.astype("float32").transpose(2, 0, 1))
                if titles:
                    title = titles[idx]
                else:
                    raise NotImplementedError
                    title = 'dog'
                # print(visual_crop.size(), idx)
                # print(visual_crop.mean(dim=(1, 2)))
                # print(orig_image.size(), idx)
                # print(orig_image.mean(dim=(1, 2)))
                inputs.append({
                    "image": orig_image,
                    "reference": visual_crop,
                    "title": title,
                    "height": heights[idx].item(),
                    "width": widths[idx].item(),
                })
            predictions = self.model(inputs, return_top_feature)
            # print(predictions)
            # exit()
            return predictions


class SiamPredictor(DefaultPredictor):
    def __call__(
        self,
        original_images: Sequence[np.ndarray],
        visual_crops: Sequence[np.ndarray],
        titles: Sequence[str] = None,
        return_top_feature: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            original_images (np.ndarray): a list of images of shape (H, W, C) (in BGR order).
            visual_crops (np.ndarray): a list of images of shape (H, W, C) (in BGR order)

        Returns:
            predictions (list[dict]):
                the output of the model for a list of images.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for idx, (original_image, visual_crop) in enumerate(zip(original_images, visual_crops)):
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                    visual_crop = visual_crop[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(
                    original_image
                )
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                reference = torch.as_tensor(
                    visual_crop.astype("float32").transpose(2, 0, 1)
                )
                if titles:
                    title = titles[idx]
                else:
                    raise NotImplementedError
                    title = 'dog'  # not used
                # print(reference.size(), idx)
                # print(reference.mean(dim=(1, 2)))
                # print(image.size(), idx)
                # print(image.mean(dim=(1, 2)))
                inputs.append(
                    {
                        "image": image,
                        "height": height,
                        "width": width,
                        "reference": reference,
                        "title": title,
                    }
                )
            # import pdb; pdb.set_trace()
            predictions = self.model(inputs, return_top_feature)
            # print(predictions)
            # exit()
            return predictions
