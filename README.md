# Simple RAG Application

A lightweight Retrieval-Augmented Generation (RAG) application using the smallest open-source models for fast, local inference.

## Features

- 🚀 **Fast inference** with llama.cpp and Qwen2.5-0.5B
- 📚 **Simple document processing** for text files
- 🔍 **Semantic search** using sentence-transformers
- 🎨 **Clean Gradio UI** for easy interaction
- 💾 **Minimal dependencies** - no heavy frameworks
- 🔢 **Pure numpy** vector operations

## Tech Stack

- **LLM**: Qwen2.5-0.5B-Instruct (GGUF Q4_K_M quantization)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Search**: Numpy cosine similarity
- **Interface**: Gradio
- **Inference**: llama.cpp

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `llama-cpp-python` may take a few minutes to install as it compiles C++ code.

### 2. Download the Model

Download the Qwen2.5-0.5B-Instruct GGUF model:

```bash
# Create models directory
mkdir -p models

# Download from HuggingFace
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf -O models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

**Alternative**: Use `curl` if `wget` is not available:

```bash
curl -L https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

**Model Size**: ~400MB

## Usage

### 1. Add Documents

Place your text files (`.txt`) in the `documents/` directory:

```bash
cp your_document.txt documents/
```

A sample document is already included for testing.

### 2. Run the Application

```bash
python app.py
```

The application will:
- Load the Qwen2.5-0.5B model
- Load the embedding model
- Process documents from the `documents/` directory
- Launch a Gradio interface at `http://127.0.0.1:7860`

### 3. Access via SSH Tunnel (Remote Server)

If running on a remote server, create an SSH tunnel:

```bash
# On your local machine
ssh -L 7860:127.0.0.1:7860 user@remote-server
```

Then open your browser to `http://localhost:7860`

### 4. Use the Interface

1. Open your browser to `http://localhost:7860`
2. Ask questions about your documents
3. Adjust parameters:
   - **Number of chunks**: How many relevant chunks to retrieve (1-5)
   - **Max tokens**: Maximum length of generated answer (50-512)
4. Click "Reload Documents" after adding new files

## Project Structure

```
rag/
├── app.py                 # Main Gradio application
├── rag_engine.py          # RAG logic (embedding, retrieval, generation)
├── document_processor.py  # Document loading and chunking
├── requirements.txt       # Python dependencies
├── models/               # GGUF model directory
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
├── documents/            # Your text files
│   └── sample.txt
└── README.md             # This file
```

## How It Works

### RAG Pipeline

1. **Document Processing**
   - Load `.txt` files from `documents/` directory
   - Split into chunks (500 chars with 50 char overlap)
   - Smart chunking at sentence boundaries

2. **Embedding**
   - Generate embeddings using sentence-transformers
   - Store in numpy arrays for fast retrieval

3. **Query Processing**
   - Embed user query
   - Calculate cosine similarity with all chunks (numpy)
   - Retrieve top-k most relevant chunks

4. **Generation**
   - Format prompt with retrieved context
   - Generate answer using Qwen2.5-0.5B via llama.cpp
   - Return answer with source citations

## Configuration

Edit `app.py` to customize:

```python
# Document chunking
DocumentProcessor(chunk_size=500, chunk_overlap=50)

# Model settings
Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Context window
    n_threads=4,       # CPU threads
    verbose=False
)

# Generation parameters
temperature=0.7        # Randomness (0.0-1.0)
top_p=0.9             # Nucleus sampling
max_tokens=256        # Max response length
```

## Performance

- **Model Load Time**: ~1-2 seconds
- **Query Response**: ~1-3 seconds (CPU)
- **Memory Usage**: ~500MB-1GB
- **Model Size**: ~400MB (Q4 quantization)

## Tips

- **Better Answers**: Use clear, specific questions
- **More Context**: Increase "Number of chunks" for complex questions
- **Longer Answers**: Increase "Max tokens" parameter
- **Document Format**: Keep documents well-structured with clear sections
- **Chunk Size**: Adjust in `document_processor.py` for your use case

## Troubleshooting

### Model Not Found
```
Error: Model not found at models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```
**Solution**: Download the model using the instructions above.

### llama-cpp-python Installation Issues
```
Error: Failed building wheel for llama-cpp-python
```
**Solution**: Ensure you have a C++ compiler installed:
- **Linux**: `sudo apt-get install build-essential`
- **macOS**: Install Xcode Command Line Tools
- **Windows**: Install Visual Studio Build Tools

### No Documents Found
```
Status: No documents found in documents/
```
**Solution**: Add `.txt` files to the `documents/` directory and click "Reload Documents".

### Out of Memory
**Solution**: Reduce `n_ctx` in `rag_engine.py` or use a smaller model.

## Extending the Application

### Add More Document Types

Modify `document_processor.py` to support PDF, DOCX, etc.:

```python
# Add PDF support
import PyPDF2

def load_pdf(filepath):
    # PDF loading logic
    pass
```

### Use Different Models

Replace the model in `app.py`:

```python
# Use a different GGUF model
MODEL_PATH = "models/your-model.gguf"
```

### Add Persistence

Save embeddings to disk to avoid recomputing:

```python
import pickle

# Save embeddings
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(rag_engine.embeddings, f)
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Qwen Team** for the Qwen2.5 models
- **llama.cpp** for fast inference
- **sentence-transformers** for embeddings
- **Gradio** for the UI framework
