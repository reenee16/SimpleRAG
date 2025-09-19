"""
LLM Client Module
Handles communication with Ollama and other LLM services
"""

import requests
import json
from typing import List, Dict, Optional


class OllamaClient:
    """Handles communication with Ollama API"""
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434"):
        """
        Initialize Ollama client
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=10
            )
            if response.status_code == 200:
                print(f"Ollama connection successful with model: {self.model_name}")
            else:
                print(f"Ollama model '{self.model_name}' not found. Available models:")
                self._list_available_models()
        except requests.exceptions.RequestException as e:
            print(f"Ollama not running or not accessible: {e}")
            print("Please start Ollama first with: ollama serve")
    
    def _list_available_models(self):
        """List available Ollama models"""
        try:
            response = requests.get(self.tags_url)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print("Could not fetch model list")
        except:
            print("Could not connect to Ollama")
    
    def generate_response(self, query: str, context_docs: List[Dict], stream: bool = True) -> str:
        """
        Generate response using Ollama
        
        Args:
            query: User query
            context_docs: List of relevant documents
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Create prompt based on query type
        prompt = self._create_prompt(query, context)
        
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
                return self._stream_response(payload)
            else:
                return self._non_stream_response(payload)
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create appropriate prompt based on query type"""
        query_lower = query.lower()
        is_total_query = any(word in query_lower for word in ['total', 'sum', 'amount', 'paid', 'cost', 'price', 'all'])
        
        if is_total_query:
            return f"""Based on the following context from multiple uploaded files, please calculate the total accurately. Make sure to include amounts from ALL files mentioned in the context.

Context from uploaded files:
{context}

Question: {query}

Instructions for total calculations:
- Look for the FINAL TOTAL AMOUNT PAID for each receipt/file
- Do NOT add breakdown items (fare, fees, etc.) separately - these are already included in the total
- Only add the main total amount from each file
- Show the calculation step by step
- Include the currency (SGD, USD, etc.)
- Make sure you don't miss any total amounts from any file

Example:
- File 1: Total Paid SGD 14.10 (use SGD 14.10)
- File 2: Total Paid SGD 22.90 (use SGD 22.90)
- File 3: Total SGD 8.50 (use SGD 8.50)
Total = SGD 14.10 + SGD 22.90 + SGD 8.50 = SGD 45.50

Answer:"""
        else:
            return f"""Based on the following context that comes from an image or file uploaded by the user, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
    
    def _stream_response(self, payload: Dict) -> str:
        """Handle streaming response"""
        response = requests.post(
            self.generate_url,
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
    
    def _non_stream_response(self, payload: Dict) -> str:
        """Handle non-streaming response"""
        response = requests.post(
            self.generate_url,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get('response', 'No response generated')
        else:
            return f"Error: {response.status_code}"
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class LLMResponseGenerator:
    """High-level response generator that combines retrieval and generation"""
    
    def __init__(self, vectorizer, llm_client):
        """
        Initialize response generator
        
        Args:
            vectorizer: Vectorizer instance
            llm_client: LLM client instance
        """
        self.vectorizer = vectorizer
        self.llm_client = llm_client
    
    def generate_response(self, query: str, top_k: int = 10, stream: bool = True) -> Dict:
        """
        Generate response for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response and sources
        """
        # Determine if this is a total query
        query_lower = query.lower()
        is_total_query = any(word in query_lower for word in ['total', 'sum', 'amount', 'paid', 'cost', 'price', 'all'])
        
        # Retrieve relevant documents
        if is_total_query:
            relevant_docs = self.vectorizer.search_for_totals(query)
        else:
            relevant_docs = self.vectorizer.search(query, top_k)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": []
            }
        
        # Generate response
        response = self.llm_client.generate_response(query, relevant_docs, stream=stream)
        
        return {
            "response": response,
            "sources": relevant_docs
        }
