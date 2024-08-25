from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTMatte(nn.Module):
    def __init__(
        self,
        *,
        backbone,
        criterion,
        pixel_mean,
        pixel_std,
        input_format,
        size_divisibility,
        decoder,
    ):
        super(ViTMatte, self).__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: Dict[str, torch.Tensor]):
        images, H, W = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images)
        outputs = self.decoder(features, images)

        outputs["phas"] = outputs["phas"][:, :, :H, :W]
        return outputs

    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        """
        Normalize, pad and batch the input images.
        """
        image = (batched_inputs["image"] - self.pixel_mean) / self.pixel_std
        trimap = batched_inputs["trimap"]

        image = torch.cat((image, trimap), dim=1)

        # Pad to ensure dimensions are multiples of 32
        B, C, H, W = image.shape
        pad_H = (32 - H % 32) % 32
        pad_W = (32 - W % 32) % 32

        if pad_H > 0 or pad_W > 0:
            new_H = (32 - H % 32) + H
            new_W = (32 - W % 32) + W
            padded_image = torch.zeros((B, C, new_H, new_W), dtype=image.dtype, device=image.device)
            padded_image[:, :, :H, :W] = image
            image = padded_image.half() if image.dtype == torch.float16 else padded_image

        return image, H, W
