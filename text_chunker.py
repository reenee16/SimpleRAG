import re
from typing import List, Dict
from pathlib import Path

try:
    import nltk
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False


class TextChunker:
    def __init__(self):
        self._setup_nltk()
    
    def _setup_nltk(self):
        if not _NLTK_OK:
            print("NLTK not available - using fallback chunking")
            return
        
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            print("NLTK resources loaded successfully")
        except Exception as e:
            print(f"NLTK setup failed: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text.strip():
            return []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # For receipt-like data, try to preserve monetary amounts in the same chunk
        monetary_patterns = re.findall(r'[A-Z]{3}\s*[\d,]+\.?\d*|[\$€£¥]\s*[\d,]+\.?\d*|total\s*:?\s*[\d,]+\.?\d*', text, re.IGNORECASE)
        
        try:
            if _NLTK_OK:
                # Split into sentences using NLTK
                sentences = nltk.sent_tokenize(text)
            else:
                raise Exception("NLTK not available")
        except Exception as e:
            print(f"NLTK sentence tokenization failed: {e}")
            print("Falling back to simple text splitting...")
            # Fallback: split by periods and other sentence endings
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if overlap > 0:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        chunked_docs = []
        
        for doc in documents:
            content = doc['content']
            source = doc['source']
            
            chunks = self.chunk_text(content, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    'content': chunk,
                    'source': f"{source} (chunk {i+1})",
                    'original_source': source,
                    'chunk_index': i,
                    'document_id': hash(source)  
                })
        
        print(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        doc_counts = {}
        for chunk in chunked_docs:
            doc_id = chunk['document_id']
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        print(f"Chunk distribution: {doc_counts}")
        return chunked_docs
    
    def chunk_by_paragraphs(self, text: str, max_chunk_size: int = 500) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_lines(self, text: str, lines_per_chunk: int = 10) -> List[str]:
        lines = text.split('\n')
        chunks = []
        
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk = '\n'.join(chunk_lines)
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
