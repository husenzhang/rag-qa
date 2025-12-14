"""
RAG Engine for embedding, retrieval, and generation.
"""
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


class RAGEngine:
    """Handles embedding, retrieval, and generation for RAG."""
    
    def __init__(self, model_path: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the RAG engine.
        
        Args:
            model_path: Path to the GGUF model file
            embedding_model: Name of the sentence-transformers model
        """
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        print("Loading LLM...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # Number of CPU threads
            verbose=False
        )
        
        self.chunks = []
        self.embeddings = None
        
    def embed_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        print(f"Embedding {len(chunks)} chunks...")
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("Embedding complete!")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((self.chunks[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate(self, query: str, context_chunks: List[Dict[str, str]], max_tokens: int = 256) -> str:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        # Format context
        context = "\n\n".join([f"[{chunk['source']}]\n{chunk['text']}" for chunk in context_chunks])
        
        # Create prompt with chain-of-thought
        prompt = f"""You are a precise information extraction assistant. Answer the question using ONLY the information in the context below.

Context:
{context}

Question: {query}

Let's solve this step by step:

Step 1: Find the relevant information in the context
- Look for the key terms from the question

Step 2: Identify which field/category it belongs to
- Check which industry or field is associated with this information

Step 3: State the answer
- Provide only the field name, nothing else

Your answer:"""
        
        # Generate response with lower temperature
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
            stop=["Question:", "\n\n\n", "Step 4"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def query(self, query: str, top_k: int = 3, max_tokens: int = 256) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'answer', 'sources', and 'retrieved_chunks'
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k)
        
        if not retrieved:
            return {
                'answer': "No documents have been loaded yet. Please add text files to the documents directory.",
                'sources': [],
                'retrieved_chunks': []
            }
        
        # Extract chunks and scores
        chunks = [item[0] for item in retrieved]
        scores = [item[1] for item in retrieved]
        
        # Generate answer
        answer = self.generate(query, chunks, max_tokens)
        
        # Format sources
        sources = []
        for chunk, score in zip(chunks, scores):
            sources.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'similarity': float(score),
                'text': chunk['text']
            })
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': len(chunks)
        }
