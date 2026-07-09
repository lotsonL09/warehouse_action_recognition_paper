# Transfer Learning for Human Action Recognition in Warehouse Environments: A comparative study of Transformer and CNN models

<div style="display: flex;">
  <img alt="background-class" src="https://github.com/user-attachments/assets/11a16688-7b01-4991-95e6-c228bfffc826" width="200" />
  <img alt="picking-class" src="https://github.com/user-attachments/assets/774ca859-2c07-419d-a824-d543882638cf"  width="200" />
  <img alt="looking-class" src="https://github.com/user-attachments/assets/325eda85-fd32-4067-bfb7-5ace2f0e44b1"     width="200"/>
  <img alt="holding-class" src="https://github.com/user-attachments/assets/33810890-36a0-45aa-9828-e4eb147eeff8" width="200" />
  <img alt="placing-class" src="https://github.com/user-attachments/assets/f6b2180d-c358-46c6-b8e8-6ee66a6bcf65"  width="200"/>
</div>

## 1. Training Dataset Structure

The model was trained using a structured dataset built from video recordings obtained from fixed surveillance cameras installed at Automatic Control System Laboratory of University of Piura. The dataset was designed to support supervised learning while remaining simple and interpretable.

### 1.1 Data Organization
Video data was organized at the directory level to reflect the training labels. Each subdirectory corresponds to a specific monitoring category.

```
dataset/
├── train/
│ ├── lifting/
│ │ ├── video_001.mp4
│ │ ├── video_002.mp4
│ │ └── ...
│ ├── placing/
│ │ ├── video_001.mp4
│ │ └── ...
│ └── holding/
│ ├── video_001.mp4
│ └── ...
│ └── looking/
│ ├── video_001.mp4
│ └── ...
│ └── background_class/
│ ├── video_001.mp4
│ └── ...
└── val/
  ├── lifting/
  ├── placing/
  └── holding/
  └── looking/
  └── background_class/
```
To increase the effective size and diversity of the training data, **data augmentation techniques** were applied during preprocessing. These augmentations introduce controlled variations in the input data, improving the model’s robustness to changes in appearance and motion patterns.

Due to **hardware limitations**, direct training on full-length videos was computationally expensive and resulted in excessively long training times. To address this constraint, **temporal sampling** was performed prior to training, selecting a fixed number of frames per video at uniform intervals. This strategy significantly reduced computational cost while preserving the temporal information required for action recognition.

This structure, combined with augmentation and sampling strategies, enables a clear separation between training and validation data while ensuring an efficient and scalable training process.

---
