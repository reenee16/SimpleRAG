import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class Vectorizer:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.dimension = None
    
    def create_index(self, documents: List[Dict]) -> None:
        print("Creating embeddings...")
        texts = [doc['content'] for doc in documents]
        self.documents = documents
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Vector index created with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10, ensure_all_documents: bool = False) -> List[Dict]:
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        if ensure_all_documents:
            return self._search_with_balanced_coverage(query_embedding, top_k)
        else:
            return self._search_normal(query_embedding, top_k)
    
    def _search_normal(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                relevant_docs.append({
                    'content': self.documents[idx]['content'],
                    'source': self.documents[idx]['source'],
                    'score': float(score),
                    'original_source': self.documents[idx].get('original_source', self.documents[idx]['source'])
                })
        
        return relevant_docs
    
    def _search_with_balanced_coverage(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        search_k = min(top_k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        doc_groups = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                doc = self.documents[idx]
                doc_id = doc.get('document_id', hash(doc['original_source']))
                
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append({
                    'content': doc['content'],
                    'source': doc['source'],
                    'score': float(score),
                    'original_source': doc['original_source']
                })
        relevant_docs = []
        for doc_id, chunks in doc_groups.items():
            best_chunk = max(chunks, key=lambda x: x['score'])
            relevant_docs.append(best_chunk)
        relevant_docs.sort(key=lambda x: x['score'], reverse=True)
        relevant_docs = relevant_docs[:top_k]
        
        print(f"Retrieved {len(relevant_docs)} chunks from {len(doc_groups)} documents")
        return relevant_docs
    
    def search_for_totals(self, query: str) -> List[Dict]:
        if self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        search_k = len(self.documents)
        print(f"Total query detected - searching ALL {search_k} chunks for complete coverage")
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        doc_groups = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                doc = self.documents[idx]
                doc_id = doc.get('document_id', hash(doc['original_source']))
                
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append({
                    'content': doc['content'],
                    'source': doc['source'],
                    'score': float(score),
                    'original_source': doc['original_source']
                })
        relevant_docs = []
        for doc_id, chunks in doc_groups.items():
            best_chunk = max(chunks, key=lambda x: x['score'])
            relevant_docs.append(best_chunk)
        total_chunks = []
        for doc_id, chunks in doc_groups.items():
            for chunk in chunks:
                chunk_lower = chunk['content'].lower()
                if any(word in chunk_lower for word in ['total', 'paid', 'amount', 'cost']):
                    total_chunks.append(chunk)
        if total_chunks:
            relevant_docs = total_chunks
            print(f"Total query: Found {len(total_chunks)} chunks with total/paid keywords")
        
        print(f"Total query: Retrieved chunks from ALL {len(doc_groups)} documents")
        return relevant_docs
    
    def get_stats(self) -> Dict:
        if self.index is None:
            return {"status": "No index created"}
        
        return {
            "status": "Index created",
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name
        }
