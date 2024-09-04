# ViTMatte for Nuke 

## Introduction

This project brings [**ViTMatte** - Boosting Image Matting with Pretrained Plain Vision Transformers](https://img.shields.io/badge/arxiv-paper-orange) to **The Foundry's Nuke**.

ViTMatte is a **natural matting neural network** that can pull high-quality **alphas** from garbage mattes *(trimaps)*. 

This implementation wraps **ViTMatte** into a single **Inference** node in Nuke, allowing it to be **easily installed** on any Nuke 13+ system running Linux or Windows.  

While **ViTMatte** works best on still images and **doesn't have temporal stability**, it can still be helpful for pulling **difficult mattes**, especially those with **fine details like hair and fur**.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-composition-1k-1)](https://paperswithcode.com/sota/image-matting-on-composition-1k-1?p=vitmatte-boosting-image-matting-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-distinctions-646)](https://paperswithcode.com/sota/image-matting-on-distinctions-646?p=vitmatte-boosting-image-matting-with)

</div>

https://github.com/rafaelperez/ViTMatte-for-Nuke/assets/1684365/bc02567a-5d95-4f5d-a8ca-45ed1e9b337c

## Compatibility

**Nuke 13.2+**, tested on **Linux** and **Windows**.

## Features

- **High quality** natural matting results
- **Fast**, less than one second per frame **(2K)**.
- **4K Support** even on 8GB GPUs.
- **Commercial use** license.
- **Overscan** and **bounding box** support
- **Easy installation** using **Nuke's Cattery** system

> [!NOTE]
> Nuke 14 and later versions are recommended. Nuke 13 has significantly slower performance due to its PyTorch version.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/ViTMatte-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**ViTMatte** will then be accessible under the toolbar at **Cattery > Segmentation > ViTMatte**.

### ⚠️ Extra Steps for Nuke 13

4. Add the path for **ViTMatte** to your `init.py`:
``` py
import nuke
nuke.pluginAddPath('./Cattery/vitmatte')
```

5. Add an menu item to the toolbar in your `menu.py`:

``` py
import nuke
toolbar = nuke.menu("Nodes")
toolbar.addCommand('Cattery/Segmentation/ViTMatte', 'nuke.createNode("vitmatte")', icon="vitmatte.png")
```
## Quick Start

**ViTMatte** creates high-quality mattes from garbage mattes in just a few clicks, using a trimap to identify edges and semi-transparent areas.

> **What is a Trimap?**  
> A trimap is a grayscale image that helps **ViTMatte** know what to focus on. Black means transparent, white means opaque, and gray means "needs work".

1. To get started, connect an image with an alpha channel to the **ViTMatte** node.

2. Use the **Edge Thickness** control to fine-tune the thickness of the trimap.

3. Adjust the **Detail** level to refine the matte.

4. If needed, disable **Fast mode** to further enhance the matte quality.

> [!NOTE]
> For a quick and easy way to create garbage mattes, check out the [SegmentAnything for Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke) tool.


## In Depth Tutorial

Comp supervisor [Alex Villabon](https://www.linkedin.com/in/avillabon/) made an **in-depth tutorial** and **demo** about VITMatte on his YouTube channel.  
Check out the [video](https://www.youtube.com/watch?v=w9sZdtiwuFE) and his channel for more awesome Nuke tips. Thanks, Alex!

[![VITMatte_demo](https://github.com/user-attachments/assets/7e729c61-7ce3-44f8-8e15-2e9465aac211)](https://www.youtube.com/watch?v=w9sZdtiwuFE
)

## Release Notes

**Latest version:** 1.2

- [x] **Memory Optimization**: Reduced peak memory usage by 3x, enabling 4K image processing on 8GB GPUs.
- [x] **Improved Gizmos Trimap**: Swapped `Dilate` for `Erode` to reduce boiling edges.

**Version 1.1**

- [x] Added *fast* model. This model is more suited for improving binary masks, like those from Segment Anything for Nuke.
- [x] Reduced memory usage with fp16 quantization
- [x] Support to Nuke 13
- [x] Improved Gizmo interface
- [x] Added overscan support
- [x] Added bounding box support
- [x] Fixed padding inference issues
- [x] Easy installation with Cattery package
- [x] New toolbar icon! ✨

## License and Acknowledgments

**ViTMatte.cat** is licensed under the MIT License, and is derived from https://github.com/hustvl/ViTMatte.

While the MIT License permits commercial use of **ViTMatte**, the dataset used for its training may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/hustvl/ViTMatte for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of RIFE.cat.**

## Citation

```
@article{yao2024vitmatte,
  title={ViTMatte: Boosting image matting with pre-trained plain vision transformers},
  author={Yao, Jingfeng and Wang, Xinggang and Yang, Shusheng and Wang, Baoyuan},
  journal={Information Fusion},
  volume={103},
  pages={102091},
  year={2024},
  publisher={Elsevier}
}
```
