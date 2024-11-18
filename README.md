# LLM Finetuning for Optimization Problems

This repository contains code for finetuning Large Language Models (LLMs) specifically for optimization problems modeling using CVXPY. The goal is to improve LLMs' ability to generate correct and efficient optimization problem formulations.

## Overview

The project focuses on:
- Finetuning various LLM models (SmolLM2, Qwen2.5-Coder) on CVXPY-specific code
- Using datasets from The Stack and GitHub Code containing optimization problems
  
## Models

Currently supported models:
- SmolLM2-360M
- SmolLM2-1.7B
- Qwen2.5-Coder-0.5B
- Qwen2.5-Coder-1.5B

## Usage

1. **Data Preparation**
```python
python data/hf_datasets.py
```

2. **Training**
```python
python train.py --config-name smollm360M  # For SmolLM2-360M
python train.py --config-name smollm1.7B   # For SmolLM2-1.7B
python train.py --config-name qwencoder0.5B # For Qwen2.5-Coder-0.5B
python train.py --config-name qwencoder1.5B # For Qwen2.5-Coder-1.5B
```

3. **Inference**
```python
python inference.py
```

## Requirements

- PyTorch
- Transformers
- PEFT
- Datasets
- Hydra
- CVXPY (for testing generated code)

## Project Structure

```
.
├── configs/           # Training configurations
├── data/             # Dataset preparation scripts
├── train.py          # Main training script
└── inference.py      # Model inference and comparison
```

## TODO

- Add CVXPY example notebooks for demonstration and testing
- Switch to instruction-tuned base models for better performance

