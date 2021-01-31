import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import albumentations as albu
import torch

class KirillSegmentModel:
    def __init__(self, model_path):
        self.model_path = model_path

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)

    def generate_mask(self, original_frame, height=256, width=320):
        frame = original_frame.copy()
        original_shape = original_frame.shape

        # ENCODER = 'resnet50'
        # ENCODER = 'efficientnet-b3'
        # ENCODER_WEIGHTS = 'imagenet'
        ENCODER = 'timm-efficientnet-b3'
        ENCODER_WEIGHTS = 'noisy-student'

        device = "cuda" if torch.cuda.is_available() else "cpu"

        img_h = height # на которых училось
        img_w = width  # на которых училось
        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=1,
            activation='sigmoid'
        )
        model.to(device)

        torch_map_location = None

        if not torch.cuda.is_available():
            torch_map_location = torch.device('cpu')

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        model = torch.load(self.model_path, map_location=torch_map_location)

        # Our operations on the frame come here
        frame_init = cv2.resize(frame, (img_w,img_h))
        # cv2.imwrite(f"frames/frame{count}.png", frame_init) - uncomment if want to save frames
        frame  = cv2.cvtColor(frame_init, cv2.COLOR_BGR2RGB)
        transform_val = albu.Compose([albu.Resize(img_h, img_w), albu.Normalize(),],)
        preprocess = self.get_preprocessing(preprocessing_fn)
        augmented = transform_val(image = frame)
        frame = augmented[ "image"]
        sample = preprocess(image=frame)
        frame = sample['image']
        x_tensor = torch.from_numpy(frame).to(device).unsqueeze(0).float()
        mask = model.predict(x_tensor)
        mask = (mask.squeeze().cpu().numpy().round())
        return mask
