"""
VGAP Training Script for Google Colab

Run this on Colab with L4/A100 GPU:
1. Mount Google Drive
2. Generate preference pairs with create_preferences.py
3. Run training with DPO

Usage (in Colab):
    !pip install transformers peft trl bitsandbytes datasets accelerate
    !python train_vgap.py --data_path /content/drive/MyDrive/RL_Project/data/dpo_train_2k.json
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def get_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Generate a unique experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments (e.g., /content/drive/MyDrive/RL_Project/experiments)
        experiment_name: Optional custom name. If None, auto-generates with timestamp.
    
    Returns:
        Full path to experiment directory
    """
    if experiment_name:
        # Use custom name, but add timestamp if directory exists to prevent overwrite
        exp_dir = os.path.join(base_dir, experiment_name)
        if os.path.exists(exp_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        # Auto-generate with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"vgap_{timestamp}")
    
    return exp_dir

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# System prompt that teaches the model the CROP format
SYSTEM_PROMPT = """You are a visual grounding assistant for web automation. Given a task description and UI element candidates with bounding boxes, output the optimal CROP region to focus on the relevant element.

The CROP region should:
- Include the target element with appropriate context
- Not be too large (avoid full page) or too small (avoid tiny crops)
- Use format: CROP(x1,y1,x2,y2) where coordinates are in pixels

Output ONLY the CROP command, nothing else."""


def format_prompt_with_chat_template(tokenizer, raw_prompt: str) -> str:
    """Format prompt using chat template for better instruction following."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt + "\n\nOutput the optimal crop region:"},
    ]
    
    # Apply chat template without generation prompt (we add chosen/rejected after)
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted


def load_preference_data(data_path: str, tokenizer=None, use_chat_format: bool = True) -> Dataset:
    """Load DPO preference pairs from JSON file."""
    print(f"Loading data from {data_path}...")
    
    with open(data_path) as f:
        pairs = json.load(f)
    
    # Format prompts with chat template if tokenizer provided
    if use_chat_format and tokenizer is not None:
        print("Applying chat template to prompts...")
        prompts = [format_prompt_with_chat_template(tokenizer, p['prompt']) for p in pairs]
    else:
        # Fallback: add simple instruction suffix
        prompts = [p['prompt'] + "\n\nOutput the optimal crop region:\nAnswer: " for p in pairs]
    
    # Convert to HuggingFace Dataset format
    dataset_dict = {
        'prompt': prompts,
        'chosen': [p['chosen'] for p in pairs],
        'rejected': [p['rejected'] for p in pairs],
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Loaded {len(dataset)} preference pairs")
    
    # Show sample
    print(f"\n--- Sample formatted prompt ---")
    print(f"Prompt (last 300 chars): ...{prompts[0][-300:]}")
    print(f"Chosen: {pairs[0]['chosen']}")
    print(f"Rejected: {pairs[0]['rejected']}")
    print("---\n")
    
    return dataset


def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2-0.5B-Instruct", use_4bit: bool = False):
    """
    Setup model with LoRA for training.
    
    Default: Full FP16/BF16 (more stable for small models)
    Optional: 4-bit quantization (use_4bit=True) for larger models
    
    Default: Qwen2-0.5B-Instruct (supports chat format)
    For better results: Qwen/Qwen2-1.5B-Instruct or Qwen/Qwen2-7B-Instruct
    """
    print(f"Loading model: {model_name}")
    print(f"Precision: {'4-bit quantized' if use_4bit else 'Full BF16'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch generation
    
    if use_4bit:
        # 4-bit quantization config (for larger models / limited VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Full precision (BF16) - more stable for small models
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
    
    # LoRA config - higher rank for full precision
    lora_config = LoraConfig(
        r=32 if not use_4bit else 16,  # Higher rank for full precision
        lora_alpha=64 if not use_4bit else 32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train_dpo(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str = "./vgap_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 1024,
):
    """Train with DPO."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nStarting DPO training...")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # DPO training config - all checkpoints saved to output_dir (on Drive)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=50,  # Save more frequently
        save_total_limit=3,  # Keep last 3 checkpoints
        remove_unused_columns=False,
        bf16=True,
        max_length=max_length,
        max_prompt_length=max_length - 128,
        report_to="none",  # Disable wandb for simplicity
        gradient_checkpointing=True,  # Explicit
        logging_dir=os.path.join(output_dir, "logs"),  # Logs on Drive too
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Save final model (LoRA adapters)
    print(f"\nSaving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save merged model for easier inference
    merged_dir = os.path.join(output_dir, "merged")
    print(f"Merging and saving full model to {merged_dir}")
    try:
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print("✓ Merged model saved!")
    except Exception as e:
        print(f"⚠ Could not save merged model: {e}")
        print("  You can still use the LoRA adapters from the main output_dir")
    
    return trainer


def generate_crop(model, tokenizer, prompt: str, max_new_tokens: int = 50, use_chat_format: bool = True) -> str:
    """Generate crop prediction from prompt."""
    # Format with chat template if the prompt isn't already formatted
    if use_chat_format and not prompt.startswith("<|"):  # Not already formatted
        formatted_prompt = format_prompt_with_chat_template(tokenizer, prompt)
    else:
        formatted_prompt = prompt
    
    # Ensure model is in eval mode
    model.eval()
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,  # Enable cache for inference
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def test_model(model, tokenizer, test_prompts: List[str]):
    """Quick test of model outputs."""
    print("\n" + "=" * 60)
    print("Testing model outputs")
    print("=" * 60)
    
    # Disable gradient checkpointing for inference
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    # Set to eval mode
    model.eval()
    
    for i, prompt in enumerate(test_prompts[:3]):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt (truncated): {prompt[:200]}...")
        
        output = generate_crop(model, tokenizer, prompt)
        print(f"Output: {output}")


def load_trained_model(model_dir: str):
    """Load a trained VGAP model for inference."""
    from peft import PeftModel
    
    print(f"Loading trained model from {model_dir}")
    
    # Get base model name from adapter config
    config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Always load tokenizer from base model (more reliable)
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if merged model exists
    merged_dir = os.path.join(model_dir, "merged")
    if os.path.exists(os.path.join(merged_dir, "model.safetensors")) or os.path.exists(os.path.join(merged_dir, "pytorch_model.bin")):
        print("Loading merged model (recommended)...")
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        print("Loading base model + LoRA adapters...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()  # Merge for faster inference
    
    model.eval()
    print(f"✓ Model loaded on {model.device}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train VGAP with DPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on 2000 examples
  python train_vgap.py \\
      --data_path /content/drive/MyDrive/RL_Project/data/dpo_train_2k.json \\
      --experiments_dir /content/drive/MyDrive/RL_Project/experiments \\
      --experiment_name vgap_2k_v1

  # Train with auto-generated experiment name
  python train_vgap.py \\
      --data_path /content/drive/MyDrive/RL_Project/data/dpo_train_2k.json \\
      --experiments_dir /content/drive/MyDrive/RL_Project/experiments

  # Evaluate trained model on test set
  python train_vgap.py \\
      --data_path /content/drive/MyDrive/RL_Project/data/dpo_test_200.json \\
      --load_model /content/drive/MyDrive/RL_Project/experiments/vgap_2k_v1
        """
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to preference pairs JSON")
    parser.add_argument("--experiments_dir", type=str, default="./experiments", help="Base directory for all experiments")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom experiment name (default: auto-generated with timestamp)")
    parser.add_argument("--output_dir", type=str, default=None, help="[DEPRECATED] Use --experiments_dir and --experiment_name instead")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Base model name (use Instruct variants)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--test_only", action="store_true", help="Only test base model, don't train")
    parser.add_argument("--no_chat_format", action="store_true", help="Disable chat template formatting")
    parser.add_argument("--load_model", type=str, default=None, help="Load trained model from this path for inference/evaluation")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (default: full BF16)")
    
    args = parser.parse_args()
    
    use_chat_format = not args.no_chat_format
    
    # Determine output directory
    if args.output_dir:
        # Legacy support for --output_dir
        output_dir = args.output_dir
        print("Note: --output_dir is deprecated. Use --experiments_dir and --experiment_name instead.")
    else:
        # New experiment versioning system
        output_dir = get_experiment_dir(args.experiments_dir, args.experiment_name)
    
    # Load raw prompts for testing
    with open(args.data_path) as f:
        raw_data = json.load(f)
    raw_prompts = [p['prompt'] for p in raw_data]
    
    print("=" * 60)
    print("VGAP Training Configuration")
    print("=" * 60)
    print(f"  Data path: {args.data_path}")
    print(f"  Samples: {len(raw_data)}")
    print(f"  Model: {args.model_name}")
    print(f"  Precision: {'4-bit' if args.use_4bit else 'BF16'}")
    if not args.load_model and not args.test_only:
        print(f"  Output dir: {output_dir}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
    print("=" * 60)
    
    # If loading a trained model for inference/evaluation
    if args.load_model:
        print(f"\nLoading trained model for evaluation...")
        model, tokenizer = load_trained_model(args.load_model)
        test_model(model, tokenizer, raw_prompts)
        print("\n✓ Done!")
        return
    
    # Setup model (need tokenizer for chat template)
    model, tokenizer = setup_model_and_tokenizer(args.model_name, use_4bit=args.use_4bit)
    
    # Load data with chat format
    dataset = load_preference_data(args.data_path, tokenizer=tokenizer, use_chat_format=use_chat_format)
    
    if args.test_only:
        # Just test the base model
        test_model(model, tokenizer, raw_prompts)
    else:
        # Train
        trainer = train_dpo(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            output_dir=output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        
        # Test after training - reload the saved model for clean inference
        print("\n" + "=" * 60)
        print("Testing trained model (loading fresh for clean inference)...")
        print("=" * 60)
        
        # Clear CUDA cache
        del model
        del trainer
        torch.cuda.empty_cache()
        
        # Load the trained model fresh
        trained_model, trained_tokenizer = load_trained_model(output_dir)
        test_model(trained_model, trained_tokenizer, raw_prompts)
    
    print("\n✓ Done!")


# Colab setup helper (legacy - see VGAP_Training_Colab.md for full guide)
COLAB_SETUP = """
# ============================================================
# VGAP Training - Quick Start
# ============================================================
# For complete guide, see: train/VGAP_Training_Colab.md

# 1. Install dependencies
!pip install -q transformers peft trl bitsandbytes datasets accelerate

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Generate training data (2000 examples)
!python /content/drive/MyDrive/RL_Project/data/create_preferences.py \\
    --split train --num_samples 2000 \\
    --output /content/drive/MyDrive/RL_Project/data/dpo_train_2k.json

# 4. Train with experiment versioning
!python /content/drive/MyDrive/RL_Project/train/train_vgap.py \\
    --data_path /content/drive/MyDrive/RL_Project/data/dpo_train_2k.json \\
    --experiments_dir /content/drive/MyDrive/RL_Project/experiments \\
    --experiment_name vgap_2k_v1 \\
    --epochs 3 --batch_size 4

# 5. Evaluate on test set
!python /content/drive/MyDrive/RL_Project/train/train_vgap.py \\
    --data_path /content/drive/MyDrive/RL_Project/data/dpo_test_200.json \\
    --load_model /content/drive/MyDrive/RL_Project/experiments/vgap_2k_v1
"""


if __name__ == "__main__":
    # Print setup instructions if running without args
    import sys
    if len(sys.argv) == 1:
        print(COLAB_SETUP)
        print("\nRun with --help for usage info")
    else:
        main()

