# QLoRA Fine-tuning for Gemma-3-4b-it

This project demonstrates QLoRA (Quantized Low-Rank Adaptation) fine-tuning of the Gemma-3-4b-it model using transformers and PEFT libraries. The model is fine-tuned to generate tool calls in response to user queries.

**System Requirements:**
- Ubuntu Linux 24.04
- Python 3.12.6
- CUDA 12.6 or 13.0 (already installed)
- NVIDIA RTX 4090 (24GB VRAM)

---

## Quick Start

### 1. Install Python 3.12.6

```bash
# Update package list
sudo apt update

# Install Python 3.12 and required build tools
sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential

# Verify installation
python3.12 --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment in .venv folder
python3.12 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Or for CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install Dependencies

```bash
# Install remaining dependencies from requirements.txt
pip install -r requirements.txt
```

### 5. Download Model

```bash
# Download Gemma-3-4b-it model (requires HuggingFace authentication)
python download_model.py
```

**Note:** You may need to authenticate with HuggingFace (see these articles for more details: https://huggingface.co/docs/huggingface_hub/quick-start#login-command and https://huggingface.co/docs/hub/security-tokens):
```bash
hf auth login
```

### 6. Run Training

```bash
# Start QLoRA fine-tuning
python train_qlora.py
```

Training will save the LoRA adapters to `./qlora_output/` directory.

### 7. Compare Models (Optional)

```bash
# Compare base model vs fine-tuned model
python compare.py
```

For detailed information about the comparison script, see [COMPARE_EXPLANATION.md](COMPARE_EXPLANATION.md).

### 8. Merge LoRA Adapters (Optional)

```bash
# Merge LoRA adapters into base model to create standalone model
python merge_qlora.py
```

This creates a merged model in `./gemma-3-4b-it-merged/` that doesn't require the base model + adapters separately. Useful for deployment.

### 9. Serve Model with vLLM (Optional)

```bash
# Serve the merged model with vLLM for fast inference
python serve_vllm.py
```

Starts an OpenAI-compatible API server on `http://0.0.0.0:8000`. Requires vLLM to be installed (`pip install vllm`).

### 10. Test vLLM API (Optional)

```bash
# Test the vLLM server with a tool call query
bash test_vllm.sh

# Or use curl directly:
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "./gemma-3-4b-it-merged", "messages": [{"role": "user", "content": "Какой у меня баланс по карте?"}], "max_tokens": 100}'
```

---

## Project Structure

```
.
├── download_model.py          # Downloads Gemma-3-4b-it model
├── train_qlora.py             # Main training script
├── compare.py                 # Compares base vs fine-tuned models
├── merge_qlora.py             # Merges LoRA adapters into base model
├── serve_vllm.py              # Serves merged model with vLLM
├── test_vllm.sh               # Test script for vLLM API
├── examples.json              # Training data (JSONL format)
├── requirements.txt           # Python dependencies
├── .venv/                     # Virtual environment (created)
├── gemma-3-4b-it/             # Downloaded model (gitignored)
├── gemma-3-4b-it-merged/      # Merged model (gitignored, created by merge_qlora.py)
└── qlora_output/              # Saved LoRA adapters (created after training)
```

---

## QLoRA Parameters

The training script uses the following QLoRA configuration (configurable in `train_qlora.py`):

**LoRA Configuration:**
- **LoRA Rank (r)**: 16 - Controls the rank of the low-rank matrices
- **LoRA Alpha**: 32 - Scaling factor for LoRA weights
- **LoRA Dropout**: 0.1 - Dropout rate for LoRA layers
- **Target Modules**: Attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

**Training Configuration:**
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 16 steps (effective batch size of 16)
- **Learning Rate**: 1e-4
- **Epochs**: 10
- **Quantization**: 4-bit NF4 with double quantization

**Memory Optimization:**
- 4-bit quantization reduces model memory footprint
- Gradient checkpointing enabled
- Mixed precision training (bfloat16 on supported GPUs)

For a detailed tutorial on QLoRA fine-tuning, see the separate QLoRA tutorial documentation.

---

## Training Data Format

The `examples.json` file contains training examples in JSONL format (one JSON object per line):

```json
{"messages": [{"role": "user", "content": "Какой у меня баланс по карте?"}, {"role": "assistant", "content": "<tool_call name=\"get_sensitive_data\" arguments=\"{}\"/>"}]}
```

Each example contains a conversation with user messages and assistant responses. The assistant responses are tool calls that the model learns to generate.

---

## Notes

- The model files (`gemma-3-4b-it/`) and virtual environment (`.venv/`) are gitignored
- Training output is saved to `qlora_output/` directory
- With RTX 4090 (24GB), you can use batch size of 1-2 depending on sequence length
- Training typically takes 10-15 minutes for 70 examples over 10 epochs
- The fine-tuned model only saves LoRA adapter weights (~130MB), not the full model

---

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `BATCH_SIZE` in `train_qlora.py`
- Reduce `MAX_SEQ_LENGTH` if using longer sequences
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size

**Model Download Issues:**
- Ensure you're logged in to HuggingFace: `huggingface-cli login`
- Check your internet connection and available disk space (~8GB needed)

**Import Errors:**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Verify all dependencies are installed: `pip list`

