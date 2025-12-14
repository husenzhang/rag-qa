"""
Simple RAG Application with Gradio UI.
"""
import os
import gradio as gr
from document_processor import DocumentProcessor
from rag_engine import RAGEngine


# Configuration
DOCUMENTS_DIR = "documents"
MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def initialize_rag():
    """Initialize the RAG system."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found at {MODEL_PATH}. Please download the model first."
    
    try:
        # Initialize components
        doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        rag_engine = RAGEngine(model_path=MODEL_PATH)
        
        # Load and process documents
        chunks = doc_processor.process_documents(DOCUMENTS_DIR)
        
        if chunks:
            rag_engine.embed_chunks(chunks)
            return rag_engine, f"Loaded {len(chunks)} chunks from {DOCUMENTS_DIR}"
        else:
            return rag_engine, f"No documents found in {DOCUMENTS_DIR}. Add .txt files and restart."
    
    except Exception as e:
        return None, f"Error initializing RAG: {str(e)}"


def format_sources(sources):
    """Format retrieved sources for display."""
    if not sources:
        return "No sources retrieved."
    
    formatted = "### Retrieved Sources:\n\n"
    for i, source in enumerate(sources, 1):
        formatted += f"**Source {i}:** {source['source']} (Chunk {source['chunk_id']}, Similarity: {source['similarity']:.3f})\n\n"
        formatted += f"```\n{source['text'][:300]}...\n```\n\n"
    
    return formatted


def query_rag(question, top_k, max_tokens):
    """Query the RAG system."""
    if rag_engine is None:
        return "RAG system not initialized. Please check the error message above.", ""
    
    if not question.strip():
        return "Please enter a question.", ""
    
    try:
        # Query the RAG system
        result = rag_engine.query(question, top_k=int(top_k), max_tokens=int(max_tokens))
        
        # Format response
        answer = result['answer']
        sources = format_sources(result['sources'])
        
        return answer, sources
    
    except Exception as e:
        return f"Error: {str(e)}", ""


def reload_documents():
    """Reload documents from the documents directory."""
    global rag_engine, status_message
    
    try:
        doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        chunks = doc_processor.process_documents(DOCUMENTS_DIR)
        
        if chunks:
            rag_engine.embed_chunks(chunks)
            status_message = f"✓ Reloaded {len(chunks)} chunks from {DOCUMENTS_DIR}"
        else:
            status_message = f"⚠ No documents found in {DOCUMENTS_DIR}"
        
        return status_message
    
    except Exception as e:
        return f"✗ Error reloading documents: {str(e)}"


# Initialize RAG system
print("Initializing RAG system...")
rag_engine, status_message = initialize_rag()
print(status_message)


# Create Gradio interface
with gr.Blocks(title="Simple RAG Application", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Simple RAG Application")
    gr.Markdown("### Powered by Qwen2.5-0.5B + llama.cpp")
    
    # Status message
    status = gr.Markdown(f"**Status:** {status_message}")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Query input
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about your documents...",
                lines=3
            )
            
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of chunks to retrieve"
                )
                max_tokens_slider = gr.Slider(
                    minimum=30,
                    maximum=256,
                    value=30,
                    step=30,
                    label="Max tokens to generate"
                )
            
            with gr.Row():
                submit_btn = gr.Button("Ask Question", variant="primary")
                reload_btn = gr.Button("Reload Documents", variant="secondary")
            
            # Answer output
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )
        
        with gr.Column(scale=1):
            # Sources output
            sources_output = gr.Markdown(label="Sources")
    
    # Instructions
    with gr.Accordion("📖 Instructions", open=False):
        gr.Markdown("""
        ### How to use:
        1. Add `.txt` files to the `documents/` directory
        2. Click "Reload Documents" to load them
        3. Ask questions about your documents
        4. Adjust retrieval and generation parameters as needed
        
        ### Tips:
        - Use clear, specific questions
        - Increase "Number of chunks" for more context
        - Increase "Max tokens" for longer answers
        
        ### Model Info:
        - **LLM:** Qwen2.5-0.5B-Instruct (GGUF Q4_K_M)
        - **Embeddings:** all-mpnet-base-v2 (768 dimensions)
        - **Vector Search:** Numpy cosine similarity
        """)
    
    # Event handlers
    submit_btn.click(
        fn=query_rag,
        inputs=[question_input, top_k_slider, max_tokens_slider],
        outputs=[answer_output, sources_output]
    )
    
    reload_btn.click(
        fn=reload_documents,
        inputs=[],
        outputs=[status]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )
