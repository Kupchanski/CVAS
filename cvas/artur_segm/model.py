import cv2
import numpy as np
from PIL import Image, ImagePalette
from artur_segm.transform import SegmentationTransform
from artur_segm.segmentation import SegmentationModule
import torch
import torch.backends.cudnn as cudnn
from functools import partial
from inplace_abn import InPlaceABN
from models.wider_resnet import WiderResNetA2
from modules import DeeplabV3

import torch.autograd.profiler as profiler

transformation = SegmentationTransform(
    2048,
    (0.41738699, 0.45732192, 0.46886091),
    (0.25685097, 0.26509955, 0.29067996),
)


def get_segmentation_wide_resnet38_deeplab_model(args):
    torch.cuda.set_device(args.rank)
    cudnn.benchmark = True
    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot(args.snapshot)
    model = SegmentationModule(body, head, 256, 65, args.fusion_mode)
    model.cls.load_state_dict(cls_state)
    model = model.cuda().eval()
    # print(model)
    return model


def load_snapshot(snapshot_file):
    """Load a training snapshot"""
    print("--- Loading model from snapshot")

    # Create network
    norm_act = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    body = WiderResNetA2([3, 3, 6, 3, 1, 1], norm_act=norm_act, dilation=(1, 2, 4, 4))
    head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # Load snapshot and recover network state
    data = torch.load(snapshot_file)
    body.load_state_dict(data["state_dict"]["body"])
    head.load_state_dict(data["state_dict"]["head"])

    return body, head, data["state_dict"]["cls"]


from datetime import datetime


def get_segmentation_by_wide_resnet38_deeplab(frame: np.ndarray, model, args) -> np.ndarray:
    # with profiler.profile(with_stack=True, profile_memory=True) as prof:
    image2 = Image.fromarray(frame)
    size = image2.size
    torch.cuda.synchronize()
    with torch.no_grad():
        img = transformation(image2.convert(mode="RGB"))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]).to(device='cuda')
        scales = [0.7, 1, 1.2]
        torch.cuda.synchronize()
        probs, preds = model(img, scales, args.flip)

        # for i, (prob, pred) in enumerate(zip(torch.unbind(probs, dim=0), torch.unbind(preds, dim=0))):
        out_size = size
        # print('start')
        # startTime = datetime.now()
        # preds = preds[0].type(torch.int16)
        # print(preds.dtype)
        torch.cuda.synchronize()
        pred = preds[0].cpu()
        # print('end')
        # print (datetime.now() - startTime )
        # print(pred.shape)

        pred_img = get_pred_image(pred, out_size, args.output_mode == "palette")

        result = np.array(pred_img)

        # out_size = size
        # preds = preds.cpu()
        # pred_img = get_pred_image(preds[0], out_size, args.output_mode == "palette")
        # result = np.array(preds)

        segment_frame = result
        mask = segment_frame == 35
        neg_mask = segment_frame != 35
        segment_frame[mask] = 255
        segment_frame[neg_mask] = 0
        cv2.imwrite("/content/test.jpg", frame)
        cv2.imwrite("/content/mask.jpg", segment_frame)
        zeros = np.zeros((out_size[1],out_size[0],3))
        zeros[:,:,0] = segment_frame
        zeros[:,:,1] = segment_frame
        zeros[:,:,2] = segment_frame
        segment_frame = zeros
        segment_frame = segment_frame.astype(int)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

    return segment_frame


def get_pred_image(tensor, out_size, with_palette):
    tensor = tensor.numpy()
    if with_palette:
        img = Image.fromarray(tensor.astype(np.uint8), mode="P")
        img.putpalette(_PALETTE)
    else:
        img = Image.fromarray(tensor.astype(np.uint8), mode="L")

    return img.resize(out_size, Image.NEAREST)


_PALETTE = np.array([[165, 42, 42],
                     [0, 192, 0],
                     [196, 196, 196],
                     [190, 153, 153],
                     [180, 165, 180],
                     [90, 120, 150],
                     [102, 102, 156],
                     [128, 64, 255],
                     [140, 140, 200],
                     [170, 170, 170],
                     [250, 170, 160],
                     [96, 96, 96],
                     [230, 150, 140],
                     [128, 64, 128],
                     [110, 110, 110],
                     [244, 35, 232],
                     [150, 100, 100],
                     [70, 70, 70],
                     [150, 120, 90],
                     [220, 20, 60],
                     [255, 0, 0],
                     [255, 0, 100],
                     [255, 0, 200],
                     [200, 128, 128],
                     [255, 255, 255],
                     [64, 170, 64],
                     [230, 160, 50],
                     [70, 130, 180],
                     [190, 255, 255],
                     [152, 251, 152],
                     [107, 142, 35],
                     [0, 170, 30],
                     [255, 255, 128],
                     [250, 0, 30],
                     [100, 140, 180],
                     [220, 220, 220],
                     [220, 128, 128],
                     [222, 40, 40],
                     [100, 170, 30],
                     [40, 40, 40],
                     [33, 33, 33],
                     [100, 128, 160],
                     [142, 0, 0],
                     [70, 100, 150],
                     [210, 170, 100],
                     [153, 153, 153],
                     [128, 128, 128],
                     [0, 0, 80],
                     [250, 170, 30],
                     [192, 192, 192],
                     [220, 220, 0],
                     [140, 140, 20],
                     [119, 11, 32],
                     [150, 0, 255],
                     [0, 60, 100],
                     [0, 0, 142],
                     [0, 0, 90],
                     [0, 0, 230],
                     [0, 80, 100],
                     [128, 64, 64],
                     [0, 0, 110],
                     [0, 0, 70],
                     [0, 0, 192],
                     [32, 32, 32],
                     [120, 10, 10]], dtype=np.uint8)
_PALETTE = np.concatenate([_PALETTE, np.zeros((256 - _PALETTE.shape[0], 3), dtype=np.uint8)], axis=0)
_PALETTE = ImagePalette.ImagePalette(palette=list(_PALETTE[:, 0]) + list(_PALETTE[:, 1]) + list(_PALETTE[:, 2]),
                                     mode="RGB")