import streamlit as st
import os
import tempfile
from pathlib import Path

try:
    import fitz  # PyMuPDF
    _PDF_OK = True
except Exception:
    _PDF_OK = False

from load_model import OllamaRAGSystem


def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    if not _PDF_OK:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install PyMuPDF")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n".join(pages)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    st.set_page_config(page_title="Simple RAG UI", page_icon="📄", layout="wide")
    st.title("📄 Simple RAG over PDFs (Ollama)")
    st.caption("Upload up to 5 PDFs, index them, and ask questions. Press Enter to submit your question.")

    # Session state
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Settings")
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

        if st.button("Initialize RAG", use_container_width=True):
            st.session_state.rag = OllamaRAGSystem(model_name=model_name, embedding_model=embed_model)
            st.success(f"Initialized with model: {model_name}")

        st.divider()

        uploads = st.file_uploader(
            "Upload PDFs (max 5)", type=["pdf"], accept_multiple_files=True
        )

        if uploads and len(uploads) > 5:
            st.warning("Please upload at most 5 PDFs. Using the first 5.")
            uploads = uploads[:5]

        if st.button("Process Documents", disabled=not (st.session_state.rag and uploads)):
            docs = []
            for up in uploads:
                try:
                    text = extract_text_from_uploaded_pdf(up)
                    if text.strip():
                        docs.append({"content": text, "source": up.name})
                except Exception as e:
                    st.error(f"Failed to read {up.name}: {e}")
            if not docs:
                st.error("No readable content found.")
            else:
                st.session_state.rag.create_vector_index(docs)
                st.session_state.indexed = True
                st.session_state.docs = docs
                st.success(f"Indexed {len(docs)} document(s).")

        # Clear chat history button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Chat area
    st.subheader("Ask a question")
    if not st.session_state.rag:
        st.info("Initialize RAG in the sidebar.")
        return
    if not st.session_state.indexed:
        st.info("Upload and process PDFs in the sidebar.")
        return

    # Chat interface with keyboard support
    with st.form(key="question_form", clear_on_submit=True):
        query = st.text_input(
            "Your question", 
            placeholder="Type your question and press Enter or click Ask...",
            key="query_input"
        )
        submit_button = st.form_submit_button("Ask")
        
        if submit_button and query.strip():
            with st.spinner("Retrieving and generating..."):
                relevant = st.session_state.rag.retrieve_relevant_docs(query, top_k=3)
                if not relevant:
                    st.warning("No relevant documents found.")
                    answer = "I couldn't find relevant information in the uploaded documents to answer your question."
                    sources = []
                else:
                    answer = st.session_state.rag.generate_response(query, relevant, stream=False)
                    sources = relevant
                
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
                st.markdown("### 🧠 Answer")
                st.write(chat['answer'])
                
                if chat['sources']:
                    st.markdown("### 📎 Sources")
                    for j, doc in enumerate(chat['sources'], 1):
                        st.write(f"{j}. {doc['source']} (score: {doc['score']:.3f})")




if __name__ == "__main__":
    main()