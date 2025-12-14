"""
Document processor for loading and chunking text files.
"""
import os
from typing import List, Dict


class DocumentProcessor:
    """Handles loading and chunking of text documents."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Load all text files from a directory.
        
        Args:
            directory: Path to directory containing text files
            
        Returns:
            List of dictionaries with 'content' and 'source' keys
        """
        documents = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating it...")
            os.makedirs(directory)
            return documents
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'content': content,
                            'source': filename
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size // 2:  # Only break if it's not too early
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Load documents and split them into chunks.
        
        Args:
            directory: Path to directory containing text files
            
        Returns:
            List of dictionaries with 'text', 'source', and 'chunk_id' keys
        """
        documents = self.load_documents(directory)
        chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc['content'])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'source': doc['source'],
                    'chunk_id': i
                })
        
        return chunks
