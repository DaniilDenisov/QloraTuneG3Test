import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./gemma-3-4b-it"  # Use local model
LORA_PATH = "./qlora_output"

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    # Check if GPU supports bfloat16
    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print("Using bfloat16 (GPU supports bf16)")
    else:
        model_dtype = torch.float16
        print("Using float16 (GPU doesn't support bf16)")
else:
    device = "cpu"
    model_dtype = torch.float32
    print("No GPU detected, using CPU with float32")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


def load_base():
    """Load base model without LoRA"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=model_dtype,
        device_map="auto"
    )
    model.eval()
    return model


def load_lora():
    """Load model with LoRA adapters"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=model_dtype,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model


def test_model(model, prompt_messages):
    """Test model with proper chat template"""
    # Use tokenizer's chat template (same as training)
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True  # Add assistant prefix
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,  # Tool call is short, don't need much
            do_sample=False,  # Deterministic for consistency
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3  # Strong repetition penalty
        )

    # Decode only the new tokens (remove input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=False  # Keep special tokens to detect tool calls
    )

    # Basic cleanup: remove end_of_turn marker if present
    if "<end_of_turn>" in generated_text:
        generated_text = generated_text.split("<end_of_turn>")[0]
    
    return generated_text.strip()


def detect_tool_call(text):
    """Detect if text contains any tool call"""
    # Simply check if <tool_call appears in the text
    return '<tool_call' in text


# Test cases that match training data
test_cases = [
    {
        "name": "Balance Query (Tool Call)",
        "messages": [{"role": "user", "content": "Какой у меня баланс по карте?"}]
    },
    {
        "name": "Transaction History (Tool Call)",
        "messages": [{"role": "user", "content": "Покажи последние операции по карте"}]
    },
    {
        "name": "Auth Check (Tool Call)",
        "messages": [{"role": "user", "content": "Я авторизован?"}]
    },
    {
        "name": "Explanation Question",
        "messages": [{"role": "user", "content": "Что такое баланс карты?"}]
    },
    {
        "name": "Greeting",
        "messages": [{"role": "user", "content": "Привет"}]
    },
    {
        "name": "Different Balance Query",
        "messages": [{"role": "user", "content": "Сколько денег осталось на карте?"}]
    }
]

print("Loading BASE model...")
base_model = load_base()

print("Loading LORA model...")
lora_model = load_lora()

print("\n" + "="*80)
print("COMPARING BASE vs LORA MODELS")
print("="*80)

# Statistics tracking
stats = {
    "base_tool_calls": 0,
    "lora_tool_calls": 0,
    "both_call_same": 0,
    "lora_learned_call": 0,
    "lora_stopped_call": 0
}

for test in test_cases:
    print(f"\n{'='*80}")
    print(f"TEST: {test['name']}")
    print(f"INPUT: {test['messages'][0]['content']}")

    base_output = test_model(base_model, test['messages'])
    lora_output = test_model(lora_model, test['messages'])

    # Detect tool calls and print them
    base_tool_call = detect_tool_call(base_output)
    lora_tool_call = detect_tool_call(lora_output)

    print(f"\nBASE MODEL OUTPUT:")
    print(base_output)
    if base_tool_call:
        print(
            f"TOOL CALL DETECTED")
    else:
        print(f"No tool call detected from base (text response)")
    print(f"\nLORA MODEL OUTPUT:")
    print(lora_output)
    if lora_tool_call:
        print(
            f"TOOL CALL DETECTED")
    else:
        print(f"No tool call detected from LoRA (text response)")

    # Check if outputs differ
    if base_output.strip() == lora_output.strip():
        print("\nOutputs are IDENTICAL")
    else:
        print("\nOutputs are DIFFERENT")

    # Check tool call behavior
    if base_tool_call and lora_tool_call:
        stats["base_tool_calls"] += 1
        stats["lora_tool_calls"] += 1
        stats["both_call_same"] += 1
    elif lora_tool_call and not base_tool_call:
        stats["lora_tool_calls"] += 1
        stats["lora_learned_call"] += 1
    elif not lora_tool_call and base_tool_call:
        stats["base_tool_calls"] += 1
        stats["lora_stopped_call"] += 1
    elif base_tool_call:
        stats["base_tool_calls"] += 1
    elif lora_tool_call:
        stats["lora_tool_calls"] += 1

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("\nTOOL CALL STATISTICS:")
print(f"  Base model tool calls: {stats['base_tool_calls']}/{len(test_cases)}")
print(f"  LORA model tool calls: {stats['lora_tool_calls']}/{len(test_cases)}")
print(f"  Both call same function: {stats['both_call_same']}")
print(f"  LORA learned to call: {stats['lora_learned_call']}")
print(f"  LORA stopped calling: {stats['lora_stopped_call']}")
