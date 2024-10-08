import logging

import torch
from torch import nn

from modeling import Detail_Capture, ViT, ViTMatte

DESTINATION = "./nuke/vitmatte_{0}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def create_vitmatte(model_size: str) -> nn.Module:
    """Create a ViTMatte model."""
    if model_size == "small":
        checkpoint_file = "./modeling/ViTMatte_S_DIS.pth"
        embed_dim = 384
        num_heads = 6
        in_chans = 384
    else:
        checkpoint_file = "./modeling/ViTMatte_B_DIS.pth"
        embed_dim = 768
        num_heads = 12
        in_chans = 768

    # Backbone parameters
    backbone_params = {
        "img_size": 512,
        "patch_size": 16,
        "in_chans": 4,
        "embed_dim": embed_dim,
        "depth": 12,
        "num_heads": num_heads,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_path_rate": 0.0,
        "norm_layer": nn.LayerNorm,
        "act_layer": nn.GELU,
        "use_abs_pos": True,
        "use_rel_pos": True,
        "rel_pos_zero_init": True,
        "window_size": 14,
        "window_block_indexes": [0, 1, 3, 4, 6, 7, 9, 10],
        "residual_block_indexes": [2, 5, 8, 11],
        "use_act_checkpoint": False,
        "pretrain_img_size": 224,
        "pretrain_use_cls_token": True,
        "out_feature": "last_feat",
        "res_conv_kernel_size": 3,
        "res_conv_padding": 1,
    }

    vitmatte_params = {
        "criterion": None,
        "pixel_mean": [123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0],
        "pixel_std": [58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0],
        "input_format": "RGB",
        "size_divisibility": 32,
        "decoder": Detail_Capture(in_chans=in_chans),
    }

    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    backbone = ViT(**backbone_params)

    vitmatte = ViTMatte(backbone=backbone, **vitmatte_params)
    vitmatte.to(device)
    vitmatte.eval()
    vitmatte.load_state_dict(checkpoint, strict=False)
    vitmatte.half()
    return vitmatte


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
    for model_size in ("small", "large"):
        LOGGER.info("Converting ViTMatte model for Nuke: %s", model_size)
        vitmatte = create_vitmatte(model_size)

        # Convert the ViTMatte model to a TorchScript model
        vitmatte_nuke = VitMatteNuke(vitmatte)
        vitmatte_traced = torch.jit.script(vitmatte_nuke)
        # LOGGER.info(vitmatte_traced)  # Uncomment to print the model

        # Save the TorchScript model
        destination = DESTINATION.format(model_size)
        vitmatte_traced.save(destination)
        LOGGER.info("Model saved to: %s", destination)
