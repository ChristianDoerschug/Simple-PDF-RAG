import streamlit as st
import os
import logging
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "llama-3.3-70b-versatile"

DEFAULT_K_RETRIEVAL = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(
    page_title="Simple PDF RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hi! Lade ein PDF hoch und stelle mir Fragen dazu. Follow-up Fragen funktionieren automatisch im gleichen Chat."
        }
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"word_count": 0, "chunk_count": 0}


@st.cache_resource
def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Cached embeddings model initialization."""
    logger.info(f"Loading embeddings model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def get_llm_model(api_key: str, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
    """Cached LLM model initialization."""
    logger.info(f"Loading LLM model: {model_name}")
    return ChatGroq(
        temperature=temperature,
        groq_api_key=api_key,
        model_name=model_name
    )


@st.cache_data
def process_pdf(uploaded_file, _embeddings) -> tuple[str, list[str], int]:
    """
    Extract and process PDF text.
    Returns: (full_text, chunks, chunk_count)
    """
    logger.info(f"Processing PDF: {uploaded_file.name}")

    pdf_reader = PdfReader(uploaded_file)
    text_parts = []

    for page_idx, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        cleaned = page_text.strip()
        if cleaned:
            text_parts.append(cleaned)

    full_text = "\n".join(text_parts)

    if not full_text:
        raise ValueError("Im PDF wurde kein lesbarer Text gefunden.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    if not chunks:
        raise ValueError("Es konnten keine Text-Chunks erstellt werden.")

    logger.info(
        f"Processed PDF: {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
    return full_text, chunks, len(chunks)


@st.cache_resource
def create_vector_store(chunks: tuple, _embeddings):
    """Cached vector store creation."""
    logger.info(f"Creating FAISS vector store with {len(chunks)} chunks")
    return FAISS.from_texts(chunks, _embeddings)


def render_sidebar():
    """Render enhanced sidebar with settings."""
    with st.sidebar:
        st.title("⚙️ Konfiguration")

        st.subheader("API Keys")
        env_groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        manual_groq_api_key = st.text_input(
            "Groq API Key (optional, wenn nicht in .env):",
            type="password"
        )
        groq_api_key = manual_groq_api_key.strip() or env_groq_api_key

        if env_groq_api_key and not manual_groq_api_key:
            st.success("✅ GROQ_API_KEY aus .env geladen")

        st.divider()

        st.subheader("RAG Parameter")
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
            help="Grösse einzelner Textblöcke"
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=DEFAULT_CHUNK_OVERLAP,
            step=50,
            help="Überlappung zwischen Chunks"
        )
        st.session_state.k_retrieval = st.slider(
            "Retrieved Documents (k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_K_RETRIEVAL,
            help="Anzahl der abgerufenen relevanten Dokumente"
        )

        st.divider()

        st.subheader("Modell Parameter")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="Kreativität des Modells (0=deterministisch, 1=kreativ)"
        )

        st.divider()

        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=st.session_state.debug_mode,
            help="Zeigt Debug-Informationen"
        )

        st.divider()
        st.write("🚀 **Powered by Groq LPU & Llama 3.3 70B Versatile**")

        return groq_api_key


def render_metrics(word_count: int, chunks_count: int):
    """Render document statistics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Wörter", word_count)
    with col2:
        st.metric("🔗 Chunks", chunks_count)
    with col3:
        st.metric("📊 Durchschn. Chunk",
                  f"{word_count // chunks_count if chunks_count > 0 else 0} Wörter")


def ensure_vector_store(uploaded_file, embeddings, debug_mode: bool = False):
    """Build vector store once per uploaded file and keep it in session state."""
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if (
        st.session_state.vector_store is not None
        and st.session_state.uploaded_file_key == current_file_key
    ):
        return

    with st.spinner("📄 Verarbeite PDF..."):
        full_text, chunks, chunk_count = process_pdf(uploaded_file, embeddings)

    with st.spinner("🧠 Erstelle Vektoren (läuft lokal)..."):
        st.session_state.vector_store = create_vector_store(
            tuple(chunks), embeddings)

    st.session_state.uploaded_file_key = current_file_key
    st.session_state.doc_stats = {
        "word_count": len(full_text.split()),
        "chunk_count": chunk_count,
    }
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": f"Dokument `{uploaded_file.name}` ist bereit. Frag mich alles dazu."
        }
    ]

    if debug_mode:
        st.info(f"[DEBUG] Indexed {chunk_count} chunks")


def run_rag_pipeline(user_question: str, groq_api_key: str, debug_mode: bool = False) -> str:
    """Execute the complete RAG pipeline."""

    try:
        try:
            docs = st.session_state.vector_store.similarity_search(
                user_question, k=st.session_state.k_retrieval)

            if debug_mode:
                st.info(f"[DEBUG] Retrieved {len(docs)} relevant documents")

            if not docs:
                st.warning(
                    "⚠️ Keine relevanten Textstellen gefunden. Bitte Frage präzisieren.")
                return None

        except Exception as exc:
            st.error(f"❌ Fehler bei der Dokumentensuche: {exc}")
            return None

        llm = get_llm_model(groq_api_key, DEFAULT_MODEL,
                            st.session_state.temperature)

        recent_messages = st.session_state.chat_messages[-6:]
        convo_lines = []
        for msg in recent_messages:
            if msg["role"] == "user":
                convo_lines.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                convo_lines.append(f"Assistant: {msg['content']}")
        chat_history = "\n".join(convo_lines)

        template = """Du bist ein hilfreicher Assistent. Nutze den Kontext aus dem Dokument und den bisherigen Chatverlauf fuer Follow-up Fragen.
    Wenn die Antwort nicht im Kontext zu finden ist, antworte: "Diese Information ist nicht im Dokument enthalten."

    Chatverlauf:
    {chat_history}

Kontext:
{context}

Frage:
{question}

Antwort:"""

        prompt = PromptTemplate.from_template(template)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        chain = (
            {
                "chat_history": lambda _: chat_history,
                "context": lambda _: context_text,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("⚡ Llama 3.3 70B Versatile liest das Dokument..."):
            try:
                response = chain.invoke(user_question)

                if debug_mode:
                    with st.expander("🔍 Debug Info"):
                        st.write("**Retrieved Documents:**")
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**Quelle {i}:**")
                            st.text(doc.page_content[:200] + "...")

                return response

            except Exception as exc:
                st.error(f"❌ Fehler bei der Modellanfrage: {exc}")
                st.info(
                    "💡 Prüfe deinen Groq API Key und ob das Modell freigeschaltet ist.")
                return None

    except Exception as exc:
        st.error(f"❌ Fehler: {exc}")
        return None


def main():
    st.title("📄 Simple PDF RAG Assistant")
    st.write(
        "Tech-Stack: Streamlit, FAISS, lokale Embeddings & Llama 3.3 70B Versatile (via Groq)")
    st.divider()

    groq_api_key = render_sidebar()

    if not groq_api_key:
        st.warning(
            "👈 Kein API Key gefunden. Lege GROQ_API_KEY in .env ab oder gib ihn links ein.")
        return

    uploaded_file = st.file_uploader("Lade ein PDF hoch", type="pdf")

    if not uploaded_file:
        st.info("📤 Bitte lade ein PDF hoch, um zu beginnen.")
        return

    embeddings = get_embeddings_model()
    ensure_vector_store(uploaded_file, embeddings,
                        debug_mode=st.session_state.debug_mode)

    st.markdown("### 💬 Chat")
    stats = st.session_state.doc_stats
    if stats["chunk_count"] > 0:
        render_metrics(stats["word_count"], stats["chunk_count"])

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(
        "Stelle eine Frage oder eine Follow-up Frage...")

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            response = run_rag_pipeline(
                user_question,
                groq_api_key,
                debug_mode=st.session_state.debug_mode
            )

            if response:
                st.markdown(response)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
