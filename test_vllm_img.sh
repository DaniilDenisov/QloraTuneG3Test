#!/bin/bash

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./gemma-3-4b-it-merged",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What do you see in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://habrastorage.org/r/w1560/getpro/habr/upload_files/743/321/7b4/7433217b4d8d3fda7824a98144fa1578.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 200
  }'