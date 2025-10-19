# RTMDet – PyTorch Implementation

![status](https://img.shields.io/badge/status-work--in--progress-yellow)

This repository is a **PyTorch port of RTMDet**, originally implemented in [MMDetection](https://github.com/open-mmlab/mmdetection).

The goal is to reimplement the network in pure PyTorch while making it possible to **load pretrained weights** from the original models.

![RTMDet-L model](assets/rtmdet-l_model_structure.jpg)

## Installation

```bash
pip install rtmdet
```

## Usage

```python
from rtmdet import RTMDet

model = RTMDet.from_preset("small")  # tiny / small / medium / large
bboxes, scores, classes = model("image.jpg")
```

## References

- **RTMDet: An Empirical Study of Real-Time Object Detectors**  
  Xiangyu Zhang, Xinyu Zhou, Zhiqi Li, et al.  
  [📄 Paper](https://arxiv.org/abs/2212.07784)

## Acknowledgments
Based on [MMDetection](https://github.com/open-mmlab/mmdetection) and the official RTMDet implementation.