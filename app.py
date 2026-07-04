import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import google.generativeai as genai
import os

st.set_page_config(page_title="RAG Doc Q&A with Gemini", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error("No Gemini API key found. Check your secrets.toml file.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.5-flash")

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_or_create_collection(name="docs", embedding_function=ef)
    return collection

collection = get_chroma_collection()

def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.title("📄 RAG Document Q&A with Gemini")
st.caption("Upload a PDF, then ask questions about it. Powered by Gemini + ChromaDB.")

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(os.urandom(8).hex())

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.processed_file:
        st.session_state.processed_file = None

    if st.button("Process Document"):
        with st.spinner("Extracting and chunking text..."):
            raw_text = extract_pdf_text(uploaded_file)
            chunks = chunk_text(raw_text)

            doc_id = f"{st.session_state.session_id}_{uploaded_file.name}"
            try:
                collection.delete(where={"source": doc_id})
            except Exception:
                pass

            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": doc_id} for _ in chunks]
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)

        st.session_state.processed_file = uploaded_file.name
        st.session_state.doc_id = doc_id
        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

st.divider()

if st.session_state.processed_file is None:
    st.info("⬆️ Please upload and process a document before asking a question.")
else:
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Searching relevant chunks..."):
            results = collection.query(
                query_texts=[query],
                n_results=4,
                where={"source": st.session_state.doc_id}
            )
            retrieved_chunks = results["documents"][0]

        if not retrieved_chunks:
            st.warning("No relevant content found. Try processing the document again.")
        else:
            context = "\n\n---\n\n".join(retrieved_chunks)

            prompt = f"""Answer the question based only on the following context from the document.
If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

            with st.spinner("Asking Gemini..."):
                response = model.generate_content(prompt)
                answer = response.text

            st.subheader("Answer")
            st.write(answer)

            with st.expander("See retrieved chunks (what Gemini used)"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk)