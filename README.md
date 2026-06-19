# potatoGPT

potatoGPT is the implementation of simplified GPT-2 architecture that is written in PyTorch. This project is designed for those who are currently learning, experimentating, and researching on their potato device. The codes are designed to be highly versatile, and easy to understand.

## Features

* Byte Pair Encoding (BPE) tokenizer training
* GPT-2 style Transformer Decoder architecture
* Multi-Head Self-Attention
* Feed Forward Networks (FFN)
* Layer Normalization
* Autoregressive text generation
* Model checkpoint saving and loading
* Configurable hyperparameters
* Training directly from raw text datasets

## Components

### Tokenizer

* Train a custom BPE tokenizer from raw text
* Save and load tokenizer vocabulary
* Convert text into token IDs and back

### Model

* Token Embeddings
* Positional Embeddings
* Multi-Head Causal Self-Attention
* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Output Projection Layer

### Training

* Training from raw text files
* Automatic train/validation split
* Periodic evaluation
* Checkpoint saving and loading
* Configurable training parameters

### Inference

* Autoregressive text generation
* Configurable context length
* Load pretrained checkpoints for generation

### Configuration 

* Edit config.py by yourself. EAFP implemented.

## Run the Code

```bash
# 1. Train tokenizer
python tokenizer_train.py

# 2. Train model
python train.py

# 3. Generate text
python inference.py
```


## Disclaimer

PotatoGPT architecture is extremely outdated compared to modern LLM's that has many new architecture and optimization techniques which includes, Multimodality processing, Mixture-of-Experts, Flash Attention, RoPE, RLHF, MoE, and others. Current implementation doesn't support DDP yet. 
