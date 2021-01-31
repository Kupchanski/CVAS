import numpy as np
import cv2
from functools import lru_cache

from utils import get_parent_dir_path
from kirill_segm.infer_model import KirillSegmentModel

@lru_cache(maxsize=10)
def get_kirill_segmentation_model():
    return KirillSegmentModel(f"{get_parent_dir_path()}/cvas/models/best_mode_lastl.pth")
    # return KirillSegmentModel(f"{get_parent_dir_path()}/cvas/models/best_model_effnet3unet.pth")
    # return KirillSegmentModel(f"{get_parent_dir_path()}/cvas/models/best_model_0.8677_unetresnet50.pth")
    # return KirillSegmentModel(f"{get_parent_dir_path()}/cvas/models/best_model_bound_loss.pth")

def get_segmentation(frame: np.ndarray, args: object) -> np.ndarray:
    model = None
    mask = None

    if args.segm == 0:
        model = get_kirill_segmentation_model()
        mask = model.generate_mask(frame)
    else:
        from artur_segm.model import get_segmentation_by_wide_resnet38_deeplab, get_segmentation_wide_resnet38_deeplab_model

        model_artur = get_segmentation_wide_resnet38_deeplab_model(args)
        mask = get_segmentation_by_wide_resnet38_deeplab(frame, model_artur, args)

    # TODO better resize
    mask = cv2.blur(mask, (5, 5))
    mask = cv2.resize(mask * 255, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.cvtColor(np.array(mask, dtype=np.uint8), cv2.COLOR_RGBA2BGR)

    return mask
