# 🧬 MTP-GAT

**MTP-GAT** is a multi-task pre-trained graph attention network for predicting the potency of endocrine-disrupting chemicals against estrogen receptor alpha (ERα). The project is based on the paper *“Predicting potency of endocrine-disrupting chemicals to estrogen receptor α: An activity cliff-aware graph attention network.”*
Endocrine-disrupting chemicals can interfere with endocrine function through interactions with receptors such as ERα. Potency prediction remains challenging because minor structural modifications can lead to substantial changes in biological activity, a phenomenon known as activity cliffs (ACs). To address this challenge, MTP-GAT incorporates two complementary pretraining tasks, namely molecular similarity learning (MolSim) and activity cliff contrastive learning (ActCliff). These tasks capture molecular structural similarity and AC-related differences. This enables the model to learn informative molecular representations for downstream potency prediction.

## ✨ Key ideas

- 🧪 **Two complementary pretraining tasks:** MolSim learns molecular similarity, while ActCliff focuses on identifying key structural features associated with substantial differences in biological activity.
- 🧩 **Activity-cliff-aware negatives:** multiple molecular representations and alkyl-chain penalties are used to identify AC-related hard negatives.
- ⚖️ **Mixture of Experts:** MolSim and ActCliff representations are combined through a learnable softmax gate for downstream prediction.

## 🗂️ Project structure

```text
├── config/
│   ├── pretrain.yaml       # Pretraining configuration
│   └── finetune.yaml       # Fine-tuning configuration
├── data/
│   ├── pretrain/           # Pretraining source data and processed cache
│   └── finetune/           # Fine-tuning data and predefined splits
├── models/                 # GAT and MoE aggregation modules
├── MTP_GAT/
│   ├── finetune/           # Fine-tuning model, dataset, and AW-NCR optimization
│   ├── pretrain/           # Pretraining models, datasets, and preprocessing
│   ├── gat/                # GAT implementation
│   └── utils/              # Molecular graph construction and utilities
└── scripts/
    ├── pretrain.py         # Pretraining 
    └── finetune.py         # Fine-tuning 
```

## ⚙️ Environment

The project was configured with **PyTorch 1.13.1 + CUDA 11.6**. Install a PyTorch Geometric build compatible with your PyTorch and CUDA versions.

Main Python dependencies:
- PyTorch and PyTorch Geometric
- RDKit
- NumPy, pandas, SciPy, and scikit-learn
- PyYAML, tqdm, and matplotlib

## 📁 Data preparation

### Pretraining data
The default `dataset.data_dir` in `config/pretrain.yaml` is `/data/pretrain`. 
The configuration refers to the MolSim file as `pretrain.csv` .
The ActCliff workflow uses data derived from `pLC50.csv`. 

### Fine-tuning data
```text
data/finetune/split-ERα/
├── train_split.csv
└── test_split.csv
```

## 🚀 Running the code
Run commands from the repository root so Python can resolve the project modules.

### 1. Pretraining
Review `config/pretrain.yaml`, 
then run:
```powershell
python scripts/pretrain.py
```

### 2. Fine-tuning
Review  `config/chembl.yaml`, 
then run:
```powershell
python scripts/finetune.py
```
