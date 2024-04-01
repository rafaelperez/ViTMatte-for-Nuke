import logging

import torch
from detectron2.checkpoint import DetectionCheckpointer
from torch import nn

from modeling import Detail_Capture, ViT, ViTMatte

# TODO: Remove dependency on detectron2
VITMATTE_MODEL = "./modeling/ViTMatte_B_DIS.pth"
DESTINATION = "./nuke/Cattery/ViTMatte/ViTMatte.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

backbone = ViT(
    img_size=512,
    patch_size=16,
    in_chans=4,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_path_rate=0.0,
    norm_layer=nn.LayerNorm,
    act_layer=nn.GELU,
    use_abs_pos=True,
    use_rel_pos=True,
    rel_pos_zero_init=True,
    window_size=14,
    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
    residual_block_indexes=[2, 5, 8, 11],
    use_act_checkpoint=False,
    pretrain_img_size=224,
    pretrain_use_cls_token=True,
    out_feature="last_feat",
    res_conv_kernel_size=3,
    res_conv_padding=1,
)

vitmatte = ViTMatte(
    backbone=backbone,
    criterion=None,
    pixel_mean=[123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0],
    pixel_std=[58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0],
    input_format="RGB",
    size_divisibility=32,
    decoder=Detail_Capture(in_chans=768),
)


class VitMatteNuke(nn.Module):
    """ViTMatte model for Nuke."""

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, :3]
        trimap = x[:, 3:4]

        image_and_trimap = {
            "image": image,
            "trimap": trimap,
        }

        return self.model(image_and_trimap)["phas"].contiguous()


if __name__ == "__main__":
    """Convert the ViTMatte model to a TorchScript model for use in Nuke."""
    # Load the ViTMatte model
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(VITMATTE_MODEL)

    # Convert the ViTMatte model to a TorchScript model
    vitmatte_nuke = VitMatteNuke(vitmatte)
    vitmatte_traced = torch.jit.script(vitmatte_nuke)
    LOGGER.info(vitmatte_traced)

    # Save the TorchScript model
    vitmatte_traced.save(DESTINATION)
    LOGGER.info("Model ViTMatte for Nuke saved to %s", DESTINATION)
