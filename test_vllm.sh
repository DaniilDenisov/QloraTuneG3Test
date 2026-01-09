#!/bin/bash
# One-line curl to test vLLM OpenAI API with tool call generation
# Model ID is "./gemma-3-4b-it-merged" (use curl http://localhost:8000/v1/models to get the model id)

curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "./gemma-3-4b-it-merged", "messages": [{"role": "user", "content": "Какой у меня баланс по карте?"}], "max_tokens": 100}'

