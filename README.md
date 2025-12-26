# Overfitting Demo Project

This project demonstrates a simple PyTorch model designed to intentionally **overfit** on a small synthetic dataset.  
The primary goal is to visualize the gap between training and testing performance and explore techniques to mitigate overfitting.

## Project Structure
- `data/`: Contains synthetic and small-scale datasets.
- `notebooks/`: Jupyter notebooks for experimentation and data visualization.
- `model/`: Stores trained model checkpoints and generated loss/accuracy plots.
- `scripts/`:
  - `train.py`: Core logic for model training, logging train/test loss, and saving metrics.
  - `evaluate.py`: Performance evaluation script for testing the model on unseen data.

## Getting Started

### 1. Environment Setup
Create and activate a dedicated Conda environment using Python 3.11:
```bash
conda create -n overfit_demo python=3.11 -y
conda activate overfit_demo