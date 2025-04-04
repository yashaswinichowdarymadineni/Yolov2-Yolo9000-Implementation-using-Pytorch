# Yolov2-Yolo9000-Implementation-using-Pytorch

## Overview
This repository contains an implementation of YOLOv2, a real-time object detection model introduced in the paper *"YOLO9000: Better, Faster, Stronger"* by Joseph Redmon and Ali Farhadi (arXiv:1612.08242). YOLOv2 improves upon its predecessor, YOLOv1, by enhancing accuracy, speed, and generalization. It achieves 78.6 mAP on VOC 2007 at 67 FPS on a Titan X GPU and can scale to over 9000 object categories with the YOLO9000 framework using hierarchical classification.

This implementation focuses on the core YOLOv2 model, including the Darknet-19 backbone, passthrough layer, and weights loading functionality, closely following the specifications in the original paper.

## Implementation Details
The implementation in this repository adheres to the YOLOv2 architecture and design choices described in the YOLO9000 paper (Section 2: "Better"). Below is a detailed breakdown of how the implementation aligns with the paper:

### Architecture
- **Input**: The model accepts images of size 416x416x3, as specified in the paper (Section 2: "High Resolution Classifier"). This resolution balances speed and accuracy, allowing the model to detect objects at various scales.
- **Backbone (Darknet-19)**: The Darknet-19 backbone is implemented with 19 convolutional layers and 5 max-pooling layers, as described in Table 1 of the paper. It downsamples the input by a factor of 32, producing a 13x13x1024 feature map. Additionally, a 26x26x512 feature map is extracted for the passthrough layer.
  - Batch normalization is applied after each convolutional layer, following the paper’s improvement (Section 2: "Batch Normalization"), which stabilizes training and improves convergence.
  - LeakyReLU activation is used with a slope of 0.1, as specified in the Darknet framework.
- **Passthrough Layer**: The passthrough (reorganization) layer is implemented to reshape the 26x26x512 feature map into 13x13x2048 by stacking adjacent features into channels (Section 2: "Fine-Grained Features"). This is concatenated with the 13x13x1024 backbone output, resulting in a 13x13x3072 feature map.
- **Output Layer**: A 1x1 convolutional layer reduces the 13x13x3072 feature map to 13x13x425, where 425 corresponds to 5 anchor boxes, each predicting 5 coordinates (x, y, w, h, objectness) and 80 class probabilities (for the COCO dataset with 80 classes). The use of anchor boxes follows the paper’s improvement (Section 2: "Dimension Clusters").
- **Anchor Boxes**: The model uses 5 anchor boxes, as determined by k-means clustering on the training data (Section 2: "Dimension Clusters"). This replaces the fully connected layers of YOLOv1, improving detection of objects with varying aspect ratios.
- **Direct Location Prediction**: The bounding box coordinates are predicted using the logistic activation function to constrain offsets between 0 and 1, as described in Section 2: "Direct Location Prediction". This stabilizes training compared to YOLOv1’s approach.

### Weights Loading
- The implementation includes functionality to load pre-trained weights (`yolov2.weights`) trained on the COCO dataset (80 classes). The weights are loaded in the order specified by the Darknet framework, ensuring compatibility with the model architecture.
- The weights file can  be downloaded from the [official YOLO website](https://pjreddie.com/darknet/yolov2/).

### Alignment with the Paper
- **Improvements Over YOLOv1**: The implementation incorporates key improvements from the paper (Section 2: "Better"):
  - Batch normalization for better convergence.
  - High-resolution input (416x416) for improved accuracy.
  - Anchor boxes for better handling of diverse object shapes.
  - Passthrough layer for fine-grained features, improving small object detection.
  - Direct location prediction for stable bounding box predictions.
- **Darknet-19 Backbone**: The backbone matches the architecture in Table 1 of the paper, with 19 convolutional layers and 5 max-pooling layers, optimized for both classification and detection tasks.
- **Performance**: While this implementation does not include training or inference, the loaded weights enable the model to achieve the performance reported in the paper: 78.6 mAP on VOC 2007 at 67 FPS on a Titan X GPU (Table 3).
- **COCO Dataset**: The model is configured for the COCO dataset (80 classes), as used in the paper for detection evaluation (Section 3: "YOLO9000").

### Limitations
- **Training**: Training is not implemented, as the focus is on using pre-trained weights. The paper describes training on ImageNet for classification and COCO for detection (Section 2: "Training for Detection"), which can be implemented in future work.
- **Evaluation**: The model’s performance (e.g., mAP on VOC or COCO) is not evaluated in this implementation, though the pre-trained weights should match the paper’s reported results.

## Files
- `YOLOv2_Implementation.ipynb`: Jupyter Notebook containing the YOLOv2 implementation, including the Darknet-19 backbone, passthrough layer, and weights loading.


## How to Run
### Requirements
- Python 3.11+
- Libraries: `numpy`, `pillow`, `torch`, `torchvision` (installed in the notebook).
- GPU recommended (e.g., NVIDIA T4) for faster computation.

