# Transformer-based Sequence Modeling for Machine Translation

This repository contains an efficient implementation of the Transformer architecture, featuring multi-head self-attention and cross-attention mechanisms, for Machine Translation applications.

## Overview

The Transformer model revolutionized the field of NLP by providing a scalable approach to sequence modeling. It leverages the self-attention mechanism to weigh the influence of different parts of the input data, making it effective for sequence modeling and tasks like machine translation.

## Multi-head Attention

The Multi-Head Attention mechanism is the main componnt of the Transformer model. It allows the model to simultaneously process information from different representation subspaces at different positions. With more heads, the model has more opportunities to focus on different parts of the input sequence, leading to richer representations and a better sequence understanding.

![Multi-head attention](images/multi_head_attention.png)

## Getting Started

Create a `Conda` environment and install the necessary packages using the following commands:

```bash
conda create -n my_env python=3.9
conda activate my_env
pip install -r requirements.txt
```

## Training the Model

To train the model, run the `train.py` script. This will automatically download the required dataset and tokenizers from HuggingFace and start the training process.
