import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path
try:
    import fitz  # PyMuPDF
    _PDF_OK = True
except Exception:
    _PDF_OK = False

class OllamaRAGSystem:
    def __init__(self, model_name="llama3.2:3b", embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize RAG system with Ollama
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "llama3.1:3b")
            embedding_model: SentenceTransformers model for embeddings
        """
        self.model_name = model_name
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        
        # Test Ollama connection
        self.test_ollama_connection()
    
    def test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=10
            )
            if response.status_code == 200:
                print(f"✅ Ollama connection successful with model: {self.model_name}")
            else:
                print(f"❌ Ollama model '{self.model_name}' not found. Available models:")
                self.list_available_models()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama not running or not accessible: {e}")
            print("Please start Ollama first with: ollama serve")
    
    def list_available_models(self):
        """List available Ollama models"""
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print("Could not fetch model list")
        except:
            print("Could not connect to Ollama")
    
    def load_documents_from_directory(self, directory_path):
        """Load text documents from a directory"""
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append({
                            'content': content,
                            'source': str(file_path)
                        })
                        print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents

    def load_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF if available."""
        if not _PDF_OK:
            raise RuntimeError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text.append(page_text)
        doc.close()
        return "\n".join(text)
    
    def create_vector_index(self, documents):
        """Create FAISS vector index from documents"""
        print("Creating embeddings...")
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        self.documents = documents
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Vector index created with {len(documents)} documents")
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                relevant_docs.append({
                    'content': self.documents[idx]['content'],
                    'source': self.documents[idx]['source'],
                    'score': float(score)
                })
        
        return relevant_docs
    
    def generate_response(self, query, context_docs, stream=True):
        """Generate response using Ollama"""
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        # Call Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 200  # Limit response length for speed
            }
        }
        
        try:
            if stream:
                response = requests.post(
                    self.ollama_url,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                token = data['response']
                                print(token, end='', flush=True)
                                full_response += token
                                
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                print()  # New line after streaming
                return full_response
            else:
                response = requests.post(
                    self.ollama_url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json().get('response', 'No response generated')
                else:
                    return f"Error: {response.status_code}"
                    
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question, stream=True, show_sources=True):
        """Main query function"""
        print(f"\n❓ Question: {question}")
        print("🔍 Retrieving relevant documents...")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question, top_k=3)
        
        if not relevant_docs:
            print("No relevant documents found.")
            return
        
        if show_sources:
            print("\n📄 Retrieved documents:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"  {i}. {doc['source']} (score: {doc['score']:.3f})")
        
        print(f"\n🤖 {self.model_name} Response:")
        print("-" * 50)
        
        # Generate response
        response = self.generate_response(question, relevant_docs, stream=stream)
        
        if not stream:
            print(response)
        
        return response

def main():
    print("🚀 Starting Ollama RAG System")
    print("=" * 50)
    
    # Initialize RAG system (use a small available model by default)
    rag = OllamaRAGSystem(model_name="llama3.2:3b")
    
    # Always use the provided PDF path
    pdf_path = r"C:\Users\Parthasarathy.Harini\Downloads\HariniParthasarathy_Resume.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Load PDF and wrap as a single document
    try:
        content = rag.load_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return
    
    documents = [{
        'content': content,
        'source': pdf_path
    }]
    
    # Create vector index
    rag.create_vector_index(documents)
    
    # Sample questions
    sample_questions = [
        "What is this person's educational background?",
        "What programming languages does this person know?",
        "What work experience does this person have?",
        "What projects has this person worked on?",
        "What are this person's key skills?"
    ]
    
    print("\n📝 Sample Questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Q&A Mode Started!")
    print("Commands: 'quit' to exit, 'samples' to show sample questions")
    print("="*50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'samples':
                print("\n📝 Sample Questions:")
                for i, q in enumerate(sample_questions, 1):
                    print(f"{i}. {q}")
                continue
            elif not question:
                continue
            
            # Process query
            rag.query(question, stream=True, show_sources=True)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()