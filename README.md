# ViTMatte for Nuke 

## Introduction

This project brings [**ViTMatte** - Boosting Image Matting with Pretrained Plain Vision Transformers](https://img.shields.io/badge/arxiv-paper-orange) to **The Foundry's Nuke**.

ViTMatte is a **natural matting neural network** that can pull high-quality **alphas** from garbage mattes *(trimaps)*. 

This implementation wraps **ViTMatte** into a single **Inference** node in Nuke, removing complicated external dependencies and allowing it to be **easily installed** on any Nuke 14+ system running Linux or Windows.

While **ViTMatte** works best on still images and **doesn't have temporal stability**, it can still be helpful for pulling **difficult mattes**, especially those with **fine details like hair and fur**.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://github.com/rafaelperez)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-composition-1k-1)](https://paperswithcode.com/sota/image-matting-on-composition-1k-1?p=vitmatte-boosting-image-matting-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-distinctions-646)](https://paperswithcode.com/sota/image-matting-on-distinctions-646?p=vitmatte-boosting-image-matting-with)

</div>

https://github.com/rafaelperez/ViTMatte-for-Nuke/assets/1684365/bc02567a-5d95-4f5d-a8ca-45ed1e9b337c

## Compatibility

**Nuke 14.0+**, tested on **Linux** and **Windows**.

## Features

- **High quality** natural matting results
- **Moderate memory requirements**, allowing **2K** and **4K** frame sizes on modern GPUs *(12GB or more)*.
- **Fast**, less than one second per frame **(2K)**.
- **Commercial use** license.
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/ViTMatte-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**ViTMatte** will then be accessible under the toolbar at **Cattery > Matting > ViTMatte**.



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
