from typing import List, Dict, Optional
from ocr_processor import OCRProcessor
from text_chunker import TextChunker
from vectorizer import Vectorizer
from llm_client import OllamaClient, LLMResponseGenerator


class RAGSystem:
    def __init__(self, 
                 model_name: str = "llama3.2:3b", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 tesseract_path: Optional[str] = None):

        self.ocr_processor = OCRProcessor(tesseract_path)
        self.text_chunker = TextChunker()
        self.vectorizer = Vectorizer(embedding_model)
        self.llm_client = OllamaClient(model_name)
        self.response_generator = LLMResponseGenerator(self.vectorizer, self.llm_client)
        
        # State
        self.documents = []
        self.indexed = False
    
    def process_documents(self, uploaded_files: List, chunk_size: int = 500, overlap: int = 50) -> Dict:

        processed_docs = []
        errors = []
        
        print(f"Processing {len(uploaded_files)} uploaded files")
        
        for i, file in enumerate(uploaded_files):
            print(f"\nProcessing file {i+1}/{len(uploaded_files)}: {file.name}")
            print(f"File type: {file.type}")
            print(f"File size: {len(file.getvalue())} bytes")
            
            try:
                # Extract text from file
                text = self.ocr_processor.extract_from_file(file)
                
                print(f"Extracted text length: {len(text)} characters")
                print(f"Text preview: {repr(text[:200])}")
                
                if text.strip():
                    processed_docs.append({
                        "content": text,
                        "source": file.name
                    })
                    print(f"Successfully processed {file.name}")
                else:
                    print(f"No text extracted from {file.name}")
                    
            except Exception as e:
                error_msg = f"Failed to read {file.name}: {e}"
                print(error_msg)
                errors.append(error_msg)
        
        print(f"\nTotal documents processed: {len(processed_docs)}")
        
        if not processed_docs:
            return {
                "success": False,
                "message": "No readable content found.",
                "errors": errors
            }

        try:
            print(f"Creating vector index for {len(processed_docs)} documents with chunking")
            chunked_docs = self.text_chunker.chunk_documents(processed_docs, chunk_size, overlap)
            self.vectorizer.create_index(chunked_docs)
            self.documents = processed_docs
            self.indexed = True
            
            return {
                "success": True,
                "message": f"Indexed {len(processed_docs)} document(s) with NLTK chunking.",
                "num_documents": len(processed_docs),
                "num_chunks": len(chunked_docs),
                "errors": errors
            }
            
        except Exception as e:
            error_msg = f"Failed to create index: {e}"
            print(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "errors": errors
            }
    
    def query(self, question: str, top_k: int = 10, stream: bool = True) -> Dict:

        if not self.indexed:
            return {
                "success": False,
                "response": "No documents indexed. Please upload and process documents first.",
                "sources": []
            }
        
        try:
            result = self.response_generator.generate_response(question, top_k, stream)
            return {
                "success": True,
                "response": result["response"],
                "sources": result["sources"]
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Error processing query: {e}",
                "sources": []
            }
    
    def get_stats(self) -> Dict:
        return {
            "indexed": self.indexed,
            "num_documents": len(self.documents),
            "ocr_available": self.ocr_processor.is_available(),
            "llm_available": self.llm_client.is_available(),
            "vectorizer_stats": self.vectorizer.get_stats()
        }
    
    def clear_index(self):
        self.documents = []
        self.indexed = False
        self.vectorizer.index = None
        self.vectorizer.documents = []
        print("Index cleared")
