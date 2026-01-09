"""
QLoRA Fine-tuning Script for Gemma-3-4b-it using examples.json
Uses transformers and peft libraries for fine-tuning
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Configuration
MODEL_PATH = "./gemma-3-4b-it"
DATA_PATH = "./examples.json"
OUTPUT_DIR = "./qlora_output"
MAX_SEQ_LENGTH = 2048

# QLoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj",
                  "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
BATCH_SIZE = 1  # Measured in examples per batch.
GRADIENT_ACCUMULATION_STEPS = 16  # Increased to maintain effective batch size. Measured in steps.
LEARNING_RATE = 1e-4  # Lower learning rate for better convergence
NUM_EPOCHS = 10  # Increased epochs for better learning
WARMUP_STEPS = 10  # Reduced warmup for small dataset. Measured in steps.
SAVE_STEPS = 50  # Save more frequently.
LOGGING_STEPS = 5  # Log output in console more frequently.


def load_data(data_path):
    """Load and format data from examples.json"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                data.append(example)
    return data


def preprocess_function(examples, tokenizer):
    """Preprocess the data for training with proper loss masking"""
    # Format each example using chat template
    texts = []
    user_texts = []  # To find where assistant response starts

    for messages in examples["messages"]:
        # Get user messages (everything before assistant)
        user_messages = [msg for msg in messages if msg["role"] != "assistant"]
        assistant_messages = [
            msg for msg in messages if msg["role"] == "assistant"]
        # Silently skipping examples without assistant messages
        if not assistant_messages:
            continue

        # Format full conversation
        full_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(full_formatted)

        # Format user part only (to determine masking position)
        if user_messages:
            user_formatted = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            user_formatted = ""
        user_texts.append(user_formatted)

    # Tokenize full conversations
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None
    )

    # Tokenize user parts to find where assistant response starts
    user_tokenized = tokenizer(
        user_texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_tensors=None
    )

    # Create labels: mask user input (set to -100), keep assistant response
    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        label = input_ids.copy()

        # Find where assistant response starts by comparing token sequences
        # This is more robust than just using length
        user_token_list = user_tokenized["input_ids"][i]

        # Find the position where user tokens end in the full sequence
        user_length = len(user_token_list)

        # Simple approach: use user_length as assistant start
        # The chat template should align correctly
        assistant_start = user_length

        # Mask all tokens before assistant response (set to -100)
        # -100 is ignored in loss computation
        for j in range(min(assistant_start, len(label))):
            label[j] = -100

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading data...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} examples")

    # Convert to dataset format (HF Dataset format differs from OpenAI json)
    dataset_dict = {"messages": [item["messages"] for item in data]}
    dataset = Dataset.from_dict(dataset_dict)

    # Preprocess dataset in batches
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split into train/val (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Check GPU availability and capabilities
    has_gpu = torch.cuda.is_available()

    # This part depends on the GPU availability and capabilities.
    if not has_gpu:
        raise RuntimeError(
            "No CUDA-compatible GPU detected. "
            "QLoRA 4-bit training requires a GPU. Aborting."
        )
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        model_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
        print("Using bfloat16 mixed precision (GPU supports bf16)")
    else:
        compute_dtype = torch.float16
        model_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
        print("Using float16 mixed precision (GPU doesn't support bf16)")

    # Configure 4-bit quantization
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=model_dtype
    )
    
    # Set pad token after model load (to match model config and avoid warnings)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # Sync with model config to avoid alignment warnings
        model.config.pad_token_id = tokenizer.pad_token_id
 
    # Disable cache for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Prepare model for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False,
        dataloader_pin_memory=has_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,  # Gradient clipping for stability
        logging_first_step=True,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Use processing_class instead of tokenizer (future-proof)
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")


if __name__ == "__main__":
    main()
