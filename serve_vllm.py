"""
Serve the merged model using vLLM
vLLM provides fast inference with continuous batching and PagedAttention

Usage:
    python serve_vllm.py
    python serve_vllm.py --model-path ./gemma-3-4b-it-merged --port 8000
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Serve model with vLLM")
    parser.add_argument("--model-path", default="./gemma-3-4b-it-merged", help="Path to merged model")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    
    args = parser.parse_args()
    
    print(f"Starting vLLM server: {args.model_path} on {args.host}:{args.port}")
    print(f"API: http://{args.host}:{args.port}/v1/chat/completions\n")
    
    # Build vLLM command with sensible defaults
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--trust-remote-code"
    ]
    
    # Run vLLM server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
