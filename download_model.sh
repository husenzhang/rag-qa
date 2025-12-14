#!/bin/bash

# Script to download Qwen2.5-0.5B-Instruct GGUF model

echo "Downloading Qwen2.5-0.5B-Instruct GGUF model..."
echo "Model size: ~400MB"
echo ""

# Create models directory if it doesn't exist
mkdir -p models

# Download the model
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
         -O models/qwen2.5-0.5b-instruct-q4_k_m.gguf
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
         -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf
else
    echo "Error: Neither wget nor curl is available."
    echo "Please install wget or curl and try again."
    exit 1
fi

# Check if download was successful
if [ -f "models/qwen2.5-0.5b-instruct-q4_k_m.gguf" ]; then
    echo ""
    echo "✓ Model downloaded successfully!"
    echo "Location: models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    echo ""
    echo "You can now run the application with: python app.py"
else
    echo ""
    echo "✗ Download failed. Please try again or download manually from:"
    echo "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    exit 1
fi
