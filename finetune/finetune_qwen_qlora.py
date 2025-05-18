import os
import json
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from peft.tuners.lora import LoraLayer
from datasets import load_dataset
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import argparse

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments for the model configuration.
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen-1_8B-Chat",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Whether to use auth token for downloading models."}
    )

@dataclass
class DataArguments:
    """
    Arguments for data configuration.
    """
    train_file: str = field(
        default="train_formatted_qa_data.jsonl",
        metadata={"help": "Path to the training data."}
    )
    validation_file: str = field(
        default="val_formatted_qa_data.jsonl",
        metadata={"help": "Path to the validation data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Training arguments for the fine-tuning.
    """
    output_dir: str = field(
        default="./qwen_finetune_output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Max gradient norm for gradient clipping."}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=200,
        metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=200,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16."}
    )
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "The optimizer to use."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing."}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "Report to wandb, tensorboard or none."}
    )

@dataclass
class LoraArguments:
    """
    Arguments for LoRA configuration.
    """
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora attention dimension."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Lora alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability of the LoRA module."}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"],
        metadata={"help": "List of module names or regex patterns to apply LoRA to."}
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use rank-stabilized LoRA."}
    )

def setup_tokenizer_for_qwen(tokenizer):
    """Setup tokenizer specifically for Qwen models."""
    # Qwen tokenizer specific setup
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Some Qwen models might need additional special tokens
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        # Set default template for chat
        tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
    
    logger.info(f"Tokenizer setup complete. Vocab size: {len(tokenizer)}")
    logger.info(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    logger.info(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    
    return tokenizer

def print_trainable_parameters(model):
    """Print trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def find_all_linear_modules(model):
    """Find all linear modules in the model."""
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    return target_modules

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with QLoRA.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-1_8B-Chat",
                        help="Path to pretrained model")
    parser.add_argument("--train_file", type=str, default="train_formatted_qa_data.jsonl",
                        help="Path to training data")
    parser.add_argument("--validation_file", type=str, default="val_formatted_qa_data.jsonl",
                        help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./qwen_finetune_output",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--find_target_modules", action="store_true",
                        help="Find all linear modules in the model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed before initializing model
    set_seed(42)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Setup tokenizer for Qwen model
    tokenizer = setup_tokenizer_for_qwen(tokenizer)

    # QLoRA 4-bit configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use nested quantization
        bnb_4bit_quant_type="nf4",  # Normalized float4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for the model
    )

    # Load model with 4-bit precision
    logger.info(f"Loading model from {args.model_name_or_path} with 4-bit quantization")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute model across available GPUs
        trust_remote_code=True,
    )

    # Extract target modules if needed
    if args.find_target_modules:
        logger.info("Finding all linear modules in the model")
        target_modules = find_all_linear_modules(model)
        logger.info(f"Found {len(target_modules)} linear modules: {target_modules}")
    else:
        # Default target modules for Qwen
        target_modules = ["c_attn", "c_proj", "w1", "w2"]
        logger.info(f"Using default target modules for Qwen: {target_modules}")

    # Resize embedding layer if needed
    if tokenizer.vocab_size != model.config.vocab_size:
        logger.info(f"Resizing embedding layer from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training")
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    logger.info("Setting up LoRA configuration")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Get PEFT model
    logger.info("Creating PEFT model with LoRA configuration")
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing
    if hasattr(model, "supports_gradient_checkpointing") and model.supports_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    else:
        logger.warning("Gradient checkpointing not supported by this model")

    # Print trainable parameters
    print_trainable_parameters(model)

    # Load datasets
    logger.info(f"Loading datasets from {args.train_file} and {args.validation_file}")
    data_files = {
        "train": args.train_file,
        "validation": args.validation_file
    }
    
    try:
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
        )
        logger.info(f"Loaded {len(raw_datasets['train'])} training examples and {len(raw_datasets['validation'])} validation examples")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        logger.info("Checking if files exist...")
        for name, file_path in data_files.items():
            if os.path.exists(file_path):
                logger.info(f"{name} file exists: {file_path}")
            else:
                logger.error(f"{name} file does not exist: {file_path}")
                logger.info(f"Working directory: {os.getcwd()}")
                logger.info(f"All files in current directory: {os.listdir('.')}")
        raise

    # Data processing function
    def preprocess_function(examples):
        # Convert to Qwen chat format
        formatted_texts = []
        for item in examples["messages"]:
            messages = item
            formatted_text = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "user":
                    formatted_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    formatted_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            formatted_texts.append(formatted_text)
        
        model_inputs = tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        
        # Create labels same as inputs (for causal LM training)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs

    # Process datasets
    logger.info("Processing datasets")
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Processing datasets",
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        logging_steps=1,
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    # Create Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Train model
    logger.info("Starting training")
    trainer.train()

    # Save model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 