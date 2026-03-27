import io
import os
import logging
import hashlib
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_K_RETRIEVAL = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_ROOT = Path(".indexes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Simple PDF RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hi! Lade ein oder mehrere PDFs hoch und stelle Fragen dazu.",
            "sources": [],
        }
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index_key" not in st.session_state:
    st.session_state.index_key = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "persist_index" not in st.session_state:
    st.session_state.persist_index = True
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {
        "word_count": 0,
        "chunk_count": 0,
        "file_count": 0,
        "page_count": 0,
    }


@st.cache_resource
def get_embeddings_model(model_name: str = EMBEDDING_MODEL):
    logger.info("Loading embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def get_llm_model(
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    logger.info("Loading LLM model: %s", model_name)
    return ChatGroq(
        temperature=temperature,
        groq_api_key=api_key,
        model_name=model_name,
    )


@st.cache_data
def extract_pdf_chunks(file_name: str, file_bytes: bytes, chunk_size: int, chunk_overlap: int):
    reader = PdfReader(io.BytesIO(file_bytes))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    texts = []
    metadatas = []
    word_count = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = raw_text.strip()
        if not cleaned:
            continue

        word_count += len(cleaned.split())
        page_chunks = splitter.split_text(cleaned)

        for chunk_idx, chunk in enumerate(page_chunks, start=1):
            value = chunk.strip()
            if not value:
                continue
            texts.append(value)
            metadatas.append(
                {
                    "source": file_name,
                    "page": page_idx,
                    "chunk": chunk_idx,
                }
            )

    if not texts:
        raise ValueError(f"{file_name}: kein lesbarer Text gefunden")

    return {
        "texts": texts,
        "metadatas": metadatas,
        "word_count": word_count,
        "page_count": len(reader.pages),
    }


def build_index_key(uploaded_files, chunk_size: int, chunk_overlap: int) -> str:
    payload_parts = [f"chunk={chunk_size}",
                     f"overlap={chunk_overlap}", f"embed={EMBEDDING_MODEL}"]
    for f in sorted(uploaded_files, key=lambda x: x.name.lower()):
        payload_parts.append(f"{f.name}:{f.size}")
    payload = "|".join(payload_parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_or_build_vector_store(uploaded_files, embeddings, chunk_size: int, chunk_overlap: int, debug_mode: bool = False, persist_index: bool = True):
    index_key = build_index_key(uploaded_files, chunk_size, chunk_overlap)
    index_dir = INDEX_ROOT / index_key
    stats_file = index_dir / "stats.json"

    if persist_index and (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        if debug_mode:
            st.info(f"[DEBUG] Index aus Datei geladen: {index_dir}")

        if stats_file.exists():
            with stats_file.open("r", encoding="utf-8") as fh:
                stats = json.load(fh)
        else:
            stats = {
                "word_count": 0,
                "chunk_count": len(store.docstore._dict),
                "file_count": len(uploaded_files),
                "page_count": 0,
            }

        return store, index_key, stats, True

    all_texts = []
    all_metas = []
    total_words = 0
    total_pages = 0

    for uploaded_file in uploaded_files:
        pdf_data = extract_pdf_chunks(
            uploaded_file.name,
            uploaded_file.getvalue(),
            chunk_size,
            chunk_overlap,
        )
        all_texts.extend(pdf_data["texts"])
        all_metas.extend(pdf_data["metadatas"])
        total_words += pdf_data["word_count"]
        total_pages += pdf_data["page_count"]

    if not all_texts:
        raise ValueError(
            "Keine Chunks erstellt. Bitte andere PDF-Dateien verwenden.")

    store = FAISS.from_texts(all_texts, embeddings, metadatas=all_metas)
    if persist_index:
        index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_dir))

    if debug_mode:
        st.info(f"[DEBUG] Neuer Index gespeichert: {index_dir}")

    stats = {
        "word_count": total_words,
        "chunk_count": len(all_texts),
        "file_count": len(uploaded_files),
        "page_count": total_pages,
    }

    if persist_index:
        with stats_file.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh)

    return store, index_key, stats, False


def render_sidebar():
    with st.sidebar:
        st.title("Konfiguration")

        st.subheader("API Keys")
        env_groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        manual_groq_api_key = st.text_input(
            "Groq API Key (optional, wenn nicht in .env):",
            type="password",
        )
        groq_api_key = manual_groq_api_key.strip() or env_groq_api_key

        if env_groq_api_key and not manual_groq_api_key:
            st.success("GROQ_API_KEY aus .env geladen")

        st.divider()

        st.subheader("RAG Parameter")
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=DEFAULT_CHUNK_OVERLAP,
            step=50,
        )
        st.session_state.k_retrieval = st.slider(
            "Retrieved Documents (k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_K_RETRIEVAL,
        )

        st.divider()

        st.subheader("Modell Parameter")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
        )

        st.divider()

        st.session_state.debug_mode = st.checkbox(
            "Debug Mode",
            value=st.session_state.debug_mode,
        )

        st.session_state.persist_index = st.checkbox(
            "Index lokal speichern",
            value=st.session_state.persist_index,
            help="Wenn deaktiviert, wird der Index nur im RAM gehalten und nicht auf Disk gespeichert.",
        )

        if st.button("Chatverlauf leeren"):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Chatverlauf wurde geleert. Du kannst direkt weiterfragen.",
                    "sources": [],
                }
            ]
            st.rerun()

        return groq_api_key, st.session_state.persist_index


def render_metrics(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dateien", stats.get("file_count", 0))
    c2.metric("Seiten", stats.get("page_count", 0))
    c3.metric("Woerter", stats.get("word_count", 0))
    c4.metric("Chunks", stats.get("chunk_count", 0))


def format_sources(docs):
    result = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unbekannt")
        page = meta.get("page", "?")
        chunk = meta.get("chunk", "?")
        snippet = " ".join(doc.page_content.split())
        result.append(
            {
                "label": f"[{idx}] {source} - Seite {page}, Chunk {chunk}",
                "snippet": snippet[:280] + ("..." if len(snippet) > 280 else ""),
            }
        )
    return result


def run_rag_pipeline(user_question: str, groq_api_key: str, debug_mode: bool = False):
    try:
        docs = st.session_state.vector_store.similarity_search(
            user_question,
            k=st.session_state.k_retrieval,
        )
    except Exception as exc:
        st.error(f"Fehler bei der Dokumentensuche: {exc}")
        return None, []

    if not docs:
        return "Keine relevanten Textstellen gefunden.", []

    llm = get_llm_model(
        groq_api_key,
        DEFAULT_MODEL,
        st.session_state.temperature,
    )

    recent_messages = st.session_state.chat_messages[-8:]
    convo_lines = []
    for msg in recent_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            convo_lines.append(f"User: {content}")
        elif role == "assistant":
            convo_lines.append(f"Assistant: {content}")
    chat_history = "\n".join(convo_lines)

    prompt = PromptTemplate.from_template(
        """Du bist ein hilfreicher Assistent fuer Dokumentfragen.
Nutze den Dokumentkontext und den Chatverlauf fuer Follow-up Fragen.
Wenn die Antwort nicht im Kontext steht, antworte klar, dass sie im Dokument nicht enthalten ist.

Chatverlauf:
{chat_history}

Kontext:
{context}

Frage:
{question}

Antwort:"""
    )

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

    with st.spinner("Antwort wird generiert..."):
        try:
            answer = chain.invoke(user_question)
            sources = format_sources(docs)
            if debug_mode:
                st.info(f"[DEBUG] {len(sources)} Quellen verwendet")
            return answer, sources
        except Exception as exc:
            st.error(f"Fehler bei der Modellanfrage: {exc}")
            return None, []


def render_chat():
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander("Quellen"):
                    for src in sources:
                        st.markdown(f"- **{src['label']}**")
                        st.caption(src["snippet"])


def main():
    st.title("Simple PDF RAG Assistant")
    st.write("Mehrere PDFs, persistenter Index, Follow-up Fragen und Quellenanzeige.")
    st.divider()

    groq_api_key, persist_index = render_sidebar()

    if not groq_api_key:
        st.warning(
            "Kein API Key gefunden. Lege GROQ_API_KEY in .env ab oder gib ihn links ein.")
        return

    uploaded_files = st.file_uploader(
        "Lade eine oder mehrere PDFs hoch",
        type="pdf",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Bitte mindestens ein PDF hochladen.")
        return

    embeddings = get_embeddings_model()

    try:
        with st.spinner("Index wird vorbereitet..."):
            vector_store, index_key, stats, loaded_from_disk = load_or_build_vector_store(
                uploaded_files,
                embeddings,
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
                st.session_state.debug_mode,
                persist_index,
            )
    except Exception as exc:
        st.error(f"Fehler beim Erstellen/Laden des Index: {exc}")
        return

    if st.session_state.index_key != index_key:
        file_names = ", ".join([f.name for f in uploaded_files])
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": f"Index bereit fuer: {file_names}",
                "sources": [],
            }
        ]

    st.session_state.vector_store = vector_store
    st.session_state.index_key = index_key
    st.session_state.doc_stats = stats

    if loaded_from_disk:
        st.caption("Index aus lokaler Speicherung geladen.")
    else:
        if persist_index:
            st.caption("Index neu erstellt und lokal gespeichert.")
        else:
            st.caption(
                "Index neu erstellt (nur im Speicher, nicht lokal gespeichert).")

    render_metrics(st.session_state.doc_stats)
    st.markdown("### Chat")
    render_chat()

    user_question = st.chat_input("Frage zum Dokumentkontext stellen...")

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question, "sources": []}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            answer, sources = run_rag_pipeline(
                user_question,
                groq_api_key,
                debug_mode=st.session_state.debug_mode,
            )
            if answer:
                st.markdown(answer)
                if sources:
                    with st.expander("Quellen"):
                        for src in sources:
                            st.markdown(f"- **{src['label']}**")
                            st.caption(src["snippet"])

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )


if __name__ == "__main__":
    main()
