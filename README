```
# GNN Baseline Integration Project

This project provides an integrated framework for evaluating and benchmarking multiple Graph Neural Network (GNN) baseline models. It supports both traditional supervised learning approaches and modern contrastive learning techniques, facilitating comprehensive experiments on node classification and link prediction tasks.

---

## Project Overview

Graph Neural Networks (GNNs) have become a cornerstone in learning from graph-structured data. This repository consolidates implementations of five representative baseline models, enabling researchers and practitioners to easily compare and extend these methods within a unified pipeline.

### Supported Models

| Model        | Learning Paradigm      | Description                                  |
|--------------|-----------------------|----------------------------------------------|
| **GraphSage**  | Traditional GNN       | Inductive node representation learning via neighborhood sampling and aggregation. |
| **GAT**       | Traditional GNN       | Graph Attention Network leveraging attention mechanisms for neighbor aggregation. |
| **GCN**       | Traditional GNN       | Graph Convolutional Network utilizing spectral graph convolutions. |
| **MVGRL**     | Contrastive Learning  | Multi-View Graph Representation Learning via contrasting node and graph views. |
| **GCA**       | Contrastive Learning  | Graph Contrastive Attention incorporating attention mechanisms into contrastive frameworks. |

---

## Directory Structure

```
./
├── Evaluation/
│   ├── evaluate_nodeclassification.py    # Node classification evaluation scripts
│   ├── evaluation.py                     # General evaluation utilities
│   ├── time_evaluation.py                # Runtime and efficiency evaluation
├── main_starter/                        # Entry points for model training and evaluation
│   ├── GCA.py
│   ├── GCONV.py
│   ├── GraphSage.py
│   ├── MVGRL.py
│   └── ...
├── models/                             # Model implementations
│   ├── GAT_Standard.py
│   ├── GCA_node.py
│   ├── GraphSage.py
│   ├── MVGRL_node.py
│   ├── Roland.py
│   └── ...
├── utils/                              # Utility modules for data processing, metrics, etc.
│   ├── data_clean.py
│   ├── data_processing.py
│   ├── metrics.py
│   ├── my_dataloader.py
│   ├── robustness_injection.py
│   └── ...
├── param/                             # Dataset-specific configuration files in JSON format
│   ├── amazon_computers.json
│   ├── coauthor_cs.json
│   └── wikics.json
├── saved_models/                      # Trained model checkpoints organized by model and dataset
│   ├── GAT/
│   ├── GCA/
│   ├── GCN/
│   └── MVGRL/
├── logs/                             # Training and evaluation logs structured by model and dataset
├── KG_predict.py                     # Knowledge graph prediction script
├── LP_Train_Contrastive_Learning.py # Link prediction training with contrastive learning
├── NC_Train_Contrastive_Learning.py # Node classification training with contrastive learning
├── README                           # This file
```

---

## Key Features

- **Unified framework** for training, evaluating, and benchmarking multiple GNN baselines.
- Supports both **traditional supervised learning** models and **contrastive learning** models.
- Modular codebase enabling easy addition of new models or datasets.
- Comprehensive **evaluation scripts**, including node classification accuracy and runtime performance.
- Logging and checkpointing for reproducibility and experiment tracking.

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric and dependencies
- Other Python packages listed in `requirements.txt` (if provided)

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

Typical workflow involves:

1. Preparing dataset parameters in the `param/` folder.
2. Starting training or evaluation via scripts in `main_starter/` or root-level scripts.
3. Evaluations are handled in the `Evaluation/` folder.
4. Logs and trained models are automatically saved in `logs/` and `saved_models/` respectively.

Example command to train GraphSage model for node classification:

```bash
python main_starter/GraphSage.py --task node_classification --dataset mooc --epochs 100
```

---

## Contributing

Contributions and suggestions are welcome. Please adhere to the existing code style and include tests for new features or models.

---

## Citation

If you use this project in your research, please cite the repository accordingly.

---

## License

Specify your license here (e.g., MIT License).

---

## Contact

For questions or support, please open an issue or contact the maintainer.

---

*This project aims to facilitate reproducible and extensible research in graph neural networks by providing a solid baseline integration for both classical and modern GNN approaches.*
```