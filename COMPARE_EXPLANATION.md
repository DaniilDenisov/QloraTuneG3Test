# Understanding compare.py

The `compare.py` script evaluates the effectiveness of QLoRA fine-tuning by comparing inference outputs from the base Gemma-3-4b-it model against the fine-tuned model with LoRA adapters. It automatically detects tool calls in model outputs and provides statistical analysis of the fine-tuning results.

---

## How the Comparison Works

### 1. Model Loading

The script loads two models for comparison:

**Base Model (`load_base()`):**
- Loads the original Gemma-3-4b-it model without any fine-tuning
- Uses the same dtype (bfloat16/float16) based on GPU capabilities
- Set to evaluation mode (`model.eval()`)

**Fine-tuned Model (`load_lora()`):**
- Loads the base Gemma-3-4b-it model
- Applies LoRA adapters from `./qlora_output/` directory using `PeftModel.from_pretrained()`
- Also set to evaluation mode for consistent comparison

Both models use the same tokenizer and are loaded with `device_map="auto"` for optimal GPU memory distribution.

### 2. Test Case Execution

The script runs a predefined set of test cases that represent different types of queries:

- **Tool Call Queries**: Questions that should trigger tool calls (e.g., "Какой у меня баланс по карте?")
- **Explanation Queries**: Questions asking for explanations (e.g., "Что такое баланс карты?")
- **General Queries**: Greetings and other general interactions

Each test case is formatted using the same chat template used during training (`tokenizer.apply_chat_template()` with `add_generation_prompt=True`), ensuring consistency between training and inference.

### 3. Generation Process

For each test case, both models generate responses using identical parameters:

**Generation Settings:**
- `max_new_tokens=80`: Limited to prevent excessive generation
- `do_sample=False`: Deterministic generation for reproducible comparisons
- `repetition_penalty=1.3`: Strong penalty to reduce repetitive text

**Output Extraction:**
- The script extracts only the newly generated tokens (excluding the input prompt)
- Removes `<end_of_turn>` markers for cleaner output
- Returns the full generated text (tool calls and all)

### 4. Tool Call Detection

The `detect_tool_call()` function uses a simple string check to identify tool calls in the generated text:

**Simple Detection:**
```python
return '<tool_call' in text
```

This approach:
- Checks if the text contains the `<tool_call` substring
- Works with any tool call format or function name
- Returns a simple boolean (`True` if found, `False` otherwise)
- Fast and reliable - no complex pattern matching needed

**Detection Results:**
- If a tool call is found: Returns `True`
- If no tool call: Returns `False`

### 5. Comparison Analysis

For each test case, the script performs several comparisons:

**Output Comparison:**
- Compares the raw text outputs from both models
- Flags if outputs are identical (which might indicate the fine-tuning didn't change behavior)
- Marks outputs as different (expected for successful fine-tuning)

**Tool Call Behavior Analysis:**
The script categorizes each test case into one of these scenarios:

1. **Both models call tool calls**: Both base and fine-tuned models generate tool calls
2. **LORA learned to call**: Fine-tuned model generates a tool call, but base model doesn't (ideal outcome)
3. **LORA stopped calling**: Base model generates a tool call, but fine-tuned model doesn't (regression)
4. **Neither calls**: Both generate text responses

### 6. Statistical Summary

At the end of all test cases, the script generates comprehensive statistics:

**Metrics Tracked:**
- **Base model tool calls**: Count of how many test cases the base model generated tool calls for
- **LORA model tool calls**: Count of how many test cases the fine-tuned model generated tool calls for
- **Both call same function**: Number of cases where both models generated tool calls
- **LORA learned to call**: Number of cases where fine-tuning successfully taught the model to call functions
- **LORA stopped calling**: Number of cases where fine-tuning removed tool call behavior (regression)

**Example Output:**
```
TOOL CALL STATISTICS:
  Base model tool calls: 0/6
  LORA model tool calls: 6/6
  Both call same function: 0
  LORA learned to call: 6
  LORA stopped calling: 0
```

This summary provides a clear picture of whether the fine-tuning achieved its goal of teaching the model to generate tool calls.

---

## Key Features

### Automatic Detection
- No manual inspection needed - the script automatically identifies tool calls using simple string matching
- Works with any tool call format - just checks for the presence of `<tool_call` in the output

### Side-by-Side Comparison
- Shows both model outputs together for easy visual comparison
- Clearly indicates whether tool calls were detected in each output

### Comprehensive Statistics
- Aggregates results across all test cases
- Provides quantitative metrics on fine-tuning success

### Consistent Testing
- Uses the same chat template as training
- Identical generation parameters for both models
- Ensures fair comparison

---

## Interpreting Results

**Successful Fine-tuning Indicators:**
- High "LORA learned to call" count
- LORA model tool calls > Base model tool calls
- Low "LORA stopped calling" count (ideally 0)

**Potential Issues:**
- "Outputs are IDENTICAL" warnings might indicate insufficient fine-tuning
- High "LORA stopped calling" suggests the model lost desired behavior
- Low tool call detection rate suggests the model didn't learn the pattern

The comparison script helps verify that QLoRA fine-tuning successfully modified the model's behavior to generate tool calls as intended, rather than producing regular conversational text responses.

