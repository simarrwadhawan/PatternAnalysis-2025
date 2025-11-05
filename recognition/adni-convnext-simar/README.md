# ConvNeXt for AD vs NC MRI Slice Classification (ADNI)

> version: 1.0    
> author: Simar Wadhawan    
> course: COMP3710 – Pattern Analysis & Recognition    
> repo_url: https://github.com/simarrawadhawan/PatternAnalysis-2025    
> branch: topic-recognition    

## Summary: 
This project fine-tunes a ConvNeXt (Small) CNN on a 2-class ADNI slice dataset (AD vs NC). It implements a clean PyTorch training/evaluation pipeline with: label smoothing, weighted loss, 384px inputs, cosine LR schedule with warmup, TTA at test-time
(orig +hflip), and threshold tuning. Final slice-level test accuracy is ~78% with strong NC recall while improving AD recall over the
baseline. All code is modularized (dataset.py, train.py, predict.py, modules.py).

### Why Convnext: 
ConvNeXt is a modern ConvNet that matches or beats transformer baselines (e.g., Swin) when trained on ViT-style design choices (depthwise   convs, GELU, LayerNorm in convnets, larger patch sizes). It remains compute-efficient and stable for grayscale medical imaging.

# Table of Contents:
  - Introduction
  - Dataset
  - Repository Structure
  - Environment & Setup
  - Training
  - Evaluation
  - Results (slice-level only)
  - Visualisations
  - Reproducibility & Determinism
  - Notes for Markers
  - References
  - License

---

# Introduction: 
  We tackle binary classification of brain MRI slices into Alzheimer’s Disease (AD) vs Normal Control (NC) using a ConvNeXt-S backbone. The   pipeline emphasizes robust training (class weights, label smoothing) and careful evaluation (TTA + calibrated threshold). We report      **slice-level** metrics per the course rubric.

## Dataset:
  - Name: ADNI (Alzheimer’s Disease Neuroimaging Initiative)
  - Task: Binary classification on 2D MRI slices: AD vs NC
  - Classes: ["AD", "NC"]
  - Approximate_counts:
    Total Images: 30520
    Train Images: 21520
    Test Images: 9000
  - Balance Note: Roughly balanced per split, mild skew per class.
    
  ### Expected Directory: 
  
    ADNI/
    ├── meta_data_with_label.json
    └── AD_NC/
        ├── train/
        │   ├── AD/    
        │   └── NC/    
        └── test/
            ├── AD/   
            └── NC/ 
            
  ### Transforms:
  
    Train: 
      - Resize to 384×384
      - RandomAffine (small degrees/translate/scale)
      - RandomHorizontalFlip(p=0.5)
      - ToTensor + Normalize (ImageNet stats for ConvNeXt pretraining)
    Test: 
      - Resize to 384×384
      - ToTensor + Normalize
    TTA: "Original + horizontal flip, logits averaged"
    
  
### Repository Structure: 
  ```text
  recognition/adni-convnext-simar/
  ├── dataset.py       # ADNIDataset + transforms
  ├── modules.py       # build_model() → ConvNeXt variants via timm
  ├── train.py         # training loop, early-stopping, checkpointing
  ├── predict.py       # evaluation + visualisations + metrics.json
  ├── utils.py         # helpers (metrics, samplers, seed utils)
  ├── requirements.txt # pinned deps
```

## Environment:
  - Python: >=3.10  
  - Requirements_file: requirements.txt  
  - Key Packages:  
    - torch, torchvision  
    - timm  
    - numpy, pandas  
    - scikit-learn  
    - matplotlib, seaborn  
    - tqdm  
    
## Installations: 
    conda create -n comp3710 python=3.10 -y
    conda activate comp3710

    # install deps
    pip install -r requirements.txt

 ## Paths:
  - Data Root: "/home/groups/comp3710/ADNI/AD_NC"
  - Checkpoint Example: "runs/adni_384_ls01_nomix_last/best.pt"
  - Image Outputs: "images/"

---

# Training:
  - Model: ConvNeXt-Small (timm)  
  - Image Size: 384  
  - Batch Size: 32  
  - Epochs: 30-60  
  - Optimizer: AdamW  
  - Scheduler: CosineAnnealingLR with warmup  
  - Learning Rate: 5.0e-5  
  - Weight Decay: 0.05  
  - Label Smoothing: 0.1  
  - Class Weights: "Computed from train split (AD vs NC)"  
  - Mixup: Disabled in final 78% run  
  - Early Stopping:  
     - Monitor: "val_acc"  
     - Patience: 10  
  - Freeze Policy: "Unfreeze last 3 stages + head (≈99.5% trainable here)"  
  
  ## Commands: 
    python train.py \
      --data-root /home/groups/comp3710/ADNI/AD_NC \
      --model convnext_small \
      --img-size 384 \
      --batch-size 32 \
      --epochs 60 \
      --lr 5e-5 \
      --weight-decay 0.05 \
      --label-smoothing 0.1 \
      --no-mixup \
      --save-dir runs/adni_384_ls01_nomix_last

---

# Evaluation:
 - Mode: Slice-level (primary, per rubric)  
 - TTA: orig + hflip (averaged)  
 - Threshold: P(NC) = 0.55 used for the 78.18% run  
  
 # Outputs: 
  
    images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── performance_metrics.png
    ├── sample_predictions.png
    ├── misclassified_samples.png
    └── metrics.json
    
  ## Commands: 
    python predict.py \
      --checkpoint runs/adni_384_ls01_nomix_last/best.pt \
      --data-root /home/groups/comp3710/ADNI/AD_NC \
      --model convnext_small \
      --batch-size 32 \
      --img-size 384 \
      --save-dir images \
      --num-workers 4
      
  - Download from Rangpur: 
    > scp -r s4977354@rangpur.compute.eait.uq.edu.au:/home/Student/s4977354/PatternAnalysis-2025/recognition/adni-convnext-simar/images ./

---

# Results Slice Level:
 ## Overall:
  - Accuracy: 0.7818
  - precision_weighted: 0.7923
  - recall_weighted: 0.7818
  - f1_weighted: 0.7795
  - specificity_AD: 0.6827
  - auc_roc: 0.8497
  - total_samples: 9000
    
 ## Per Class:
  - AD:
    - Precision: 0.8472
    - Recall: 0.6827
    - f1: 0.7561
    - Support: 4460
      
  - NC:
    - Precision: 0.7383
    - Recall: 0.8791
    - f1: 0.8025
    - Support: 4540
      
  - Confusion Matrix Counts:
    - TN_AD_correct: 3045
    - FP_AD_as_NC: 1415
    - FN_NC_as_AD: 549
    - TP_NC_correct: 3991
      
  Note: 
  > Metrics above correspond to TTA (orig+hflip) with threshold P(NC)=0.55.
    This configuration yielded the best slice-level balance in our runs.

---

# Figures:
  
    - caption: "Confusion Matrix (slice-level)"
      path: "images/confusion_matrix.png"
      
    - caption: "ROC Curve (slice-level)"
      path: "images/roc_curve.png"
    - caption: "Performance Metrics (slice-level)"
      path: "images/performance_metrics.png"
    - caption: "Sample Predictions"
      path: "images/sample_predictions.png"
    - caption: "Misclassified Samples"
      path: "images/misclassified_samples.png"

---

# Usage Quickstart: 

  ### 1) Install
  conda activate comp3710
  pip install -r requirements.txt

  ### 2) Train
  python train.py --data-root /home/groups/comp3710/ADNI/AD_NC --img-size 384 --batch-size 32

  ### 3) Evaluate (slice-level + figures)
  python predict.py --checkpoint runs/adni_384_ls01_nomix_last/best.pt --save-dir images

  ### 4) Copy figures off Rangpur
  scp -r s4977354@rangpur.compute.eait.uq.edu.au:/home/Student/s4977354/PatternAnalysis-2025/recognition/adni-convnext-simar/images ./

## Checklist To Submit: 
  - [x] Push updated .py files (no checkpoints)
  - [x] README with slice-level 78% results and figures
  - [x] Open PR from topic-recognition → main with clear title/body
  - [x] Export README to PDF (GitHub “Print to PDF” or Markdown → PDF)
  - [x] Submit PDF + repo link

---

# References:
  - Name: ConvNeXt  
    Link: "https://arxiv.org/abs/2201.03545"
  - Name: timm: PyTorch Image Models  
    Link: "https://github.com/huggingface/pytorch-image-models"
  - Name: ADNI overview  
    Link: "https://adni.loni.usc.edu/"
  - Name: ROC/AUC interpretation (medical)  
    Link: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12260203/"

License: Apache-2.0 (same as course starter)
