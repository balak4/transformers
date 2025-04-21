# Greedy Learning Rate Scheduler Pre-training Scripts

This directory contains scripts for pre-training causal language models (LLMs) from scratch using the Greedy Learning Rate scheduler and other schedulers like Cosine with both single-GPU and distributed multi-GPU training support.

## Overview

The scripts implement pre-training for different model architectures:
- `pre-train-gpt2.py`: Pre-training for GPT-2 models
- `pre-train-llama3.2-1b.py`: Pre-training for LLaMA 3.2 1B models with support for both greedy and cosine schedulers
- `pretrain-llm-gpt2.ipynb`: Jupyter notebook demonstrating data preprocessing and training workflow

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate 0.20+
- Datasets 2.10+
- DeepSpeed (for multi-GPU training)

### Recommended Hardware

- **Single GPU training**: At least 16GB VRAM
- **Multi-GPU training**: 2+ GPUs with 16GB+ VRAM each
- For LLaMA 3.2 1B: At least 24GB VRAM total (can be distributed across GPUs)

## Dataset Preparation

The scripts use a pre-tokenized dataset of code snippets. The dataset should be prepared using the steps outlined in the `pretrain-llm-gpt2.ipynb` notebook. The default location for the datasets is:

```
./logs/codeparrot-ds/tokensized_dataset_train_50K_seed_42/
```

## Pre-training Options

### Single GPU Training

Train on a single GPU with simplified setup. This is the default mode and works well for smaller models like GPT-2.

### Multi-GPU Training with DeepSpeed

For larger models like LLaMA 3.2 1B, you can use multi-GPU training with DeepSpeed's Zero Redundancy Optimizer (ZeRO) Stage 2. This distributes the model optimization state across multiple GPUs, allowing for efficient training of larger models.

## Usage Instructions

### GPT-2 Pre-training

```bash
# Single GPU
python3 pre-train-gpt2.py

# Multi-GPU with DeepSpeed (must specify --mode multi)
cd examples/greedy-lr
torchrun --nproc_per_node=2 pre-train-gpt2.py --mode multi
```

### LLaMA 3.2 1B Pre-training

```bash
# Single GPU with cosine scheduler (default)
python3 pre-train-llama3.2-1b.py

# Single GPU with greedy scheduler
python3 pre-train-llama3.2-1b.py --lr_scheduler greedy

# Multi-GPU with DeepSpeed (2 GPUs) and cosine scheduler
cd examples/greedy-lr
torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine

# Multi-GPU with DeepSpeed (2 GPUs) and greedy scheduler
torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler greedy

# Multi-GPU with 4 or 8 GPUs
torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
torchrun --nproc_per_node=8 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
```

### Command-line Arguments

The scripts support the following command-line arguments:

- `--mode`: Training mode
  - `single`: Single GPU training (default)
  - `multi`: Multi-GPU training with DeepSpeed
- `--lr_scheduler`: Learning rate scheduler type
  - `cosine`: Cosine learning rate scheduler (default)
  - `greedy`: Greedy learning rate scheduler
- `--num_gpus`: Number of GPUs to use in multi-GPU mode (default: 2)
- `--local_rank`: Local rank for distributed training (automatically set by torchrun)

## Scaling Tips

### Batch Size and Gradient Accumulation

The default settings are:
- `per_device_train_batch_size`: 4 (LLaMA) / 48 (GPT-2)
- `gradient_accumulation_steps`: 32 (LLaMA) / 8 (GPT-2)

For larger GPU setups, you can increase batch size or decrease gradient accumulation:

```python
# For 8x H100 setup
training_args_dict = {
    "per_device_train_batch_size": 8,  # Double the batch size
    "gradient_accumulation_steps": 16,  # Half the accumulation steps
    # other args...
}
```

### Precision Options

The scripts default to FP32 training for stability. To enable mixed precision:

1. For BF16 (recommended for NVIDIA Ampere+ GPUs):
```python
# In DeepSpeed config
"bf16": {"enabled": True},
```

2. For FP16 (older GPUs):
```python
# In DeepSpeed config
"fp16": {"enabled": True},
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing (already enabled in scripts)

2. **Distributed Training Errors**:
   - Ensure all processes can access the dataset
   - Verify unique DeepSpeed config paths per process
   - Check for port conflicts with `--master_port` option for torchrun

3. **Numerical Stability Issues**:
   - Start with FP32 training
   - Gradually transition to mixed precision once stable
   - Use DeepSpeed ZeRO Stage 1 instead of 2 if needed

### DeepSpeed Configuration

The DeepSpeed configuration can be adjusted in the `get_deepspeed_config` function:

```python
def get_deepspeed_config(num_gpus, batch_size, grad_accum_steps):
    return {
        # Adjust settings here
        "zero_optimization": {
            "stage": 2,  # Can try 1 if having issues
            # other settings...
        },
        # These are set to "auto" to avoid batch size conflicts
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }
```

## Outputs

The pre-training outputs are saved to:
```
./logs/codeparrot-ds/{MODEL_NAME}/run{run_num}/{run_name}/{date}/
```

This directory contains:
- `tensorboard/`: TensorBoard logs
- `output/`: Checkpoints and training outputs
- `model/`: Final saved model
