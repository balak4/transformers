import os
import torch
import torch.nn as nn
import argparse
import json
import datetime
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    LlamaConfig,
    LlamaForCausalLM
)
from datasets import load_from_disk
import logging

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA pre-training with flexible GPU options")
    parser.add_argument(
        "--mode", 
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Training mode: 'single' for single GPU, 'multi' for multi-GPU with DeepSpeed"
    )
    parser.add_argument(
        "--num_gpus", 
        type=int,
        default=2,
        help="Number of GPUs to use in multi-GPU mode (default: 2)"
    )
    parser.add_argument(
        "--lr_scheduler", 
        type=str,
        default="cosine",
        choices=["greedy", "cosine"],
        help="Learning rate scheduler: 'greedy' for Greedy LR or 'cosine' for Cosine LR"
    )
    parser.add_argument(
        "--local_rank", 
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher"
    )
    return parser.parse_args()

# Generate DeepSpeed config based on parameters
def get_deepspeed_config(num_gpus, batch_size, grad_accum_steps):
    """Generate DeepSpeed config dict based on training parameters."""
    return {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_training(model_name):
    # Set random seed
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prepared dataset
    dataset_name = 'tokensized_dataset_train_50K_seed_42'
    load_dir = f'./logs/codeparrot-ds/{dataset_name}/'
    
    # Check if dataset exists
    if not os.path.exists(load_dir):
        logger.error(f"Dataset not found at {load_dir}. Please ensure the dataset is prepared.")
        raise FileNotFoundError(f"Dataset directory {load_dir} does not exist")
        
    tokenized_datasets = load_from_disk(load_dir)
    
    logger.info(f"Dataset columns: {tokenized_datasets['train'].column_names}")
    
    # Model configuration
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=2048,        # Appropriate for 1B parameter model
        intermediate_size=5632,  # Standard for LLaMA 3.2 1B
        num_hidden_layers=22,    # Appropriate for 1B parameter model
        num_attention_heads=16,  # Standard for this size
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,       # LLaMA 3 specific 
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        initializer_range=0.02,  # Important for stability
        use_cache=False,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,  # Start with fp32, convert later if needed
    )
    
    # Initialize model with config
    try:
        logger.info(f"Initializing LLaMA model with config: {config}")
        model = LlamaForCausalLM(config)

        for layer in model.model.layers:
            # Common practice for transformer architectures
            nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=config.initializer_range)

        # Log model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    
    return model, tokenizer, tokenized_datasets

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configure GPU usage based on mode
    if args.mode == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        use_deepspeed = False
        logger.info(f"Running in single-GPU mode")
    else:
        # For multi-GPU, let torchrun/distributed launch handle GPU assignment
        use_deepspeed = True
        # Don't set CUDA_VISIBLE_DEVICES in multi-GPU mode, as it's managed by the launcher
        logger.info(f"Running in multi-GPU mode with local_rank: {args.local_rank}")
    # Print environment information safely
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # In multi-GPU mode, each process should only access its assigned GPU
    if args.mode == "multi":
        # When running with torchrun, each process should only access its assigned device
        if torch.cuda.is_available() and args.local_rank >= 0:
            # Set device based on local_rank
            device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(args.local_rank)
            logger.info(f"Process {args.local_rank} using device: {device}")
            
            # Clear GPU cache for this device only
            torch.cuda.empty_cache()
            logger.info(f"GPU {args.local_rank} memory allocated: {torch.cuda.memory_allocated(args.local_rank) / 1024**3:.2f} GB")
    else:
        # Single GPU mode - safe to query all devices
        if torch.cuda.is_available():
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_properties(0).name}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            logger.info(f"GPU 0 memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"GPU 0 memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    
    # Setup
    model, tokenizer, tokenized_datasets = setup_training(model_name=MODEL_NAME)
    
    # Create output directories
    exp_name = "codeparrot-ds"
    run_num = 'test1'
    run_name = args.lr_scheduler
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    base_dir = f"./logs/{exp_name}/{MODEL_NAME}/run{run_num}/{run_name}/{date}"
    logging_dir = f"{base_dir}/tensorboard"
    output_dir = f"{base_dir}/output"
    model_dir = f"{base_dir}/model"
    
    logger.info(f"logs directory: {logging_dir}")
    
    for dir_path in [logging_dir, output_dir, model_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Training arguments with conditional DeepSpeed setup
    training_args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "logging_dir": logging_dir,
        "logging_steps": 10,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": 32,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "evaluation_strategy": "steps",
        "eval_steps": 1_000,
        "warmup_steps": 1_000,
        "save_steps": 500,
        "save_total_limit": 3,
        "report_to": "tensorboard",
        "lr_scheduler_type": run_name
    }
    
    # Add common parameters
    training_args_dict.update({
        "remove_unused_columns": False,
        "debug": "underflow_overflow",
        "logging_first_step": True
    })
    
    # Add greedy LR specific parameters only when using greedy scheduler
    if run_name == "greedy":
        training_args_dict.update({
            "min_lr": 1.85e-5,
            "smooth": True,
            "factor": 0.95
        })
    
    if use_deepspeed:
        # For multi-GPU training, add DeepSpeed config and DistributedDataParallel settings
        # Use the num_gpus argument directly in multi-GPU mode
        num_gpus = args.num_gpus if args.mode == "multi" else 1
        ds_config = get_deepspeed_config(
            num_gpus=num_gpus,
            batch_size=training_args_dict["per_device_train_batch_size"],
            grad_accum_steps=training_args_dict["gradient_accumulation_steps"]
        )
        
        # Create a temporary DeepSpeed config file with unique name per process
        ds_config_path = os.path.join(os.path.dirname(output_dir), f"ds_config_rank{args.local_rank}.json")
        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
            
        training_args_dict["deepspeed"] = ds_config_path
        # Use the parsed local_rank argument instead of environment variable
        training_args_dict["local_rank"] = args.local_rank
        
        # Add DDP parameters for multi-GPU training
        training_args_dict["ddp_find_unused_parameters"] = True
        
        logger.info(f"DeepSpeed config written to {ds_config_path}")
        
    training_args = TrainingArguments(**training_args_dict)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer - simplified for both single and multi-GPU modes
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator
    )
    
    if use_deepspeed:
        logger.info("Using DeepSpeed for distributed training")
    else:
        # For single GPU, explicitly move model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(model, "to"):
            model.to(device)
        logger.info(f"Model moved to device: {device}")
    
    # Disable cache for training
    if isinstance(model, torch.nn.DataParallel):
        model.module.config.use_cache = False
    else:
        model.config.use_cache = False
    
    # Train
    logger.info(f"Starting training with {MODEL_NAME} using {run_name} LR Scheduler...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {model_dir}")
    if isinstance(model, torch.nn.DataParallel):
        trainer.model.module.save_pretrained(
            model_dir,
            safe_serialization=False
        )
    else:
        trainer.model.save_pretrained(
            model_dir,
            safe_serialization=False
        )

if __name__ == "__main__":
    """
    Run modes:
    
    1. Single GPU (default uses cosine scheduler):
       python3 pre-train-llama3.2-1b.py
       python3 pre-train-llama3.2-1b.py --mode single --lr_scheduler cosine
       
    2. Single GPU with greedy scheduler:
       python3 pre-train-llama3.2-1b.py --lr_scheduler greedy
       
    3. Multi-GPU with DeepSpeed (works with 2, 4, or 8 GPUs):
       # From within the examples/greedy-lr directory with 2 GPUs:
       torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With 4 GPUs:
       torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With 8 GPUs:
       torchrun --nproc_per_node=8 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With greedy scheduler:
       torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler greedy
    """
    main()
