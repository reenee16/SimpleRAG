
import streamlit as st
from rag_system import RAGSystem


def main():
    st.set_page_config(page_title="PDF-AI", page_icon="📄", layout="wide")
    st.title("PDF-AI")
    st.caption("Upload up to 5 PDFs or images and ask questions")
    
    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Settings")
        
        # Model settings
        model_name = st.text_input("Ollama model", value="llama3.2:3b")
        embed_model = st.selectbox(
            "Embedding model",
            [
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
            ],
            index=0,
        )
        
        # Chunking settings
        st.subheader("Text Chunking Settings")
        chunk_size = st.slider("Chunk size (characters)", min_value=200, max_value=1000, value=500, step=50)
        overlap = st.slider("Overlap (characters)", min_value=0, max_value=200, value=50, step=10)
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=20, value=10, step=1)
        st.caption("Higher values ensure better coverage across all uploaded files")

        # Initialize RAG system
        if st.button("Start", use_container_width=True):
            try:
                st.session_state.rag_system = RAGSystem(
                    model_name=model_name,
                    embedding_model=embed_model
                )
                st.success(f"Initialized with model: {model_name}")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")

        st.divider()

        # File upload
        uploads = st.file_uploader(
            "Upload PDFs or Images (max 5)", 
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"], 
            accept_multiple_files=True
        )

        if uploads and len(uploads) > 5:
            st.warning("Please upload at most 5 files. Using the first 5.")
            uploads = uploads[:5]

        # Process documents
        if st.button("Process Documents", disabled=not (st.session_state.rag_system and uploads)):
            if st.session_state.rag_system:
                with st.spinner("Processing documents..."):
                    result = st.session_state.rag_system.process_documents(
                        uploads, 
                        chunk_size=chunk_size, 
                        overlap=overlap
                    )
                
                if result["success"]:
                    st.session_state.indexed = True
                    st.success(result["message"])
                    if result.get("errors"):
                        for error in result["errors"]:
                            st.warning(error)
                else:
                    st.error(result["message"])
                    if result.get("errors"):
                        for error in result["errors"]:
                            st.error(error)

        # Clear chat history
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # System stats
        if st.session_state.rag_system:
            st.divider()
            st.subheader("System Status")
            stats = st.session_state.rag_system.get_stats()
            
            if stats["indexed"]:
                st.success("Documents indexed")
            else:
                st.info("No documents indexed")
            
            if stats["ocr_available"]:
                st.success("OCR available")
            else:
                st.warning("OCR not available")
            
            if stats["llm_available"]:
                st.success("LLM available")
            else:
                st.warning("LLM not available")
            
            if stats["indexed"]:
                st.info(f"{stats['num_documents']} documents, {stats['vectorizer_stats']['num_documents']} chunks")

    # Main chat area
    st.subheader("Ask a question")
    
    if not st.session_state.rag_system:
        st.info("Click 'Start' in the sidebar to initialize the RAG system.")
        return
    
    if not st.session_state.indexed:
        st.info("Upload and process documents in the sidebar.")
        return

    # Chat interface
    with st.form(key="question_form", clear_on_submit=True):
        query = st.text_input(
            "Your question", 
            placeholder="Type your question and press Enter or click Ask...",
            key="query_input"
        )
        submit_button = st.form_submit_button("Ask")
        
        if submit_button and query.strip():
            with st.spinner("Retrieving and generating..."):
                result = st.session_state.rag_system.query(query, top_k=top_k, stream=False)
                
                if result["success"]:
                    answer = result["response"]
                    sources = result["sources"]
                else:
                    answer = result["response"]
                    sources = []
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": sources
                })
                st.rerun()

    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Chat History")
        
        # Display conversations in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}", expanded=(i == 0)):
                st.markdown("### Answer")
                st.write(chat['answer'])
                
                if chat['sources']:
                    st.markdown("### Sources")
                    for j, doc in enumerate(chat['sources'], 1):
                        st.write(f"{j}. {doc['source']} (score: {doc['score']:.3f})")


if __name__ == "__main__":
    main()
