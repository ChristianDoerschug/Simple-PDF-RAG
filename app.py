import os
import logging
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    get_embeddings_model,
    get_llm_model,
    update_or_load_vector_store,
    infer_response_language,
    get_rag_chain
)

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
def cached_get_embeddings_model(model_name: str = EMBEDDING_MODEL):
    return get_embeddings_model(model_name)


@st.cache_resource
def cached_get_llm_model(
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    return get_llm_model(api_key, model_name, temperature)


def render_sidebar():
    with st.sidebar:
        st.title("Konfiguration")

        st.subheader("Kurs-Verwaltung")
        existing_courses = [d.name for d in INDEX_ROOT.iterdir() if d.is_dir()]
        course_options = ["-- Neuer Kurs --"] + existing_courses
        selected_course = st.selectbox(
            "Lernfach / Kurs wählen", course_options)

        if selected_course == "-- Neuer Kurs --":
            course_name = st.text_input("Neuer Kursname:", "Mein Kurs")
        else:
            course_name = selected_course

        st.divider()

        st.subheader("Lern-Modus")
        st.session_state.learning_mode = st.radio(
            "Wähle deinen Modus aus:",
            ["Standard Chat",
                "Quiz-Master (Prüfung)", "Sokratisch (Erklären)"],
            help="Bestimmt, wie die KI auf deine Eingaben reagiert."
        )

        st.divider()

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

        return course_name, groq_api_key, st.session_state.persist_index


def render_metrics(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dateien", stats.get("file_count", 0))
    c2.metric("Seiten", stats.get("page_count", 0))
    c3.metric("Woerter", stats.get("word_count", 0))
    c4.metric("Chunks", stats.get("chunk_count", 0))


def format_sources(docs):
    result = []
    # Dedupliziere Quellen nach Seite und Dokument, um übersichtlicher zu bleiben
    seen = set()
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unbekannt")
        page = meta.get("page", "?")

        # Einzigartige ID für die Quelle
        doc_id = f"{source}_page_{page}"

        if doc_id not in seen:
            seen.add(doc_id)
            snippet = " ".join(doc.page_content.split())
            # Kürzerer Anzeigename
            short_source = source.replace(".pdf", "")
            result.append(
                {
                    "label": f"📄 {short_source} – Folie {page}",
                    "snippet": snippet[:280] + ("..." if len(snippet) > 280 else ""),
                }
            )
    return result


def infer_response_language(text: str) -> str:
    lower = f" {text.lower()} "
    german_markers = [" der ", " die ", " und ", " ist ",
                      " nicht ", " ich ", " bitte ", "frage", "dokument"]
    english_markers = [" the ", " and ", " is ", " not ",
                       " please ", "question", "document", "what", "how", "can you"]

    german_score = sum(1 for token in german_markers if token in lower)
    english_score = sum(1 for token in english_markers if token in lower)

    if german_score > english_score:
        return "de"
    return "en"


def run_rag_pipeline(user_question: str, groq_api_key: str, debug_mode: bool = False):
    response_lang = infer_response_language(user_question)
    response_language_name = "German" if response_lang == "de" else "English"

    try:
        docs = st.session_state.vector_store.similarity_search(
            user_question,
            k=st.session_state.k_retrieval,
        )
    except Exception as exc:
        st.error(f"Fehler bei der Dokumentensuche: {exc}")
        return None, []

    if not docs:
        if response_lang == "de":
            return "Keine relevanten Textstellen gefunden.", []
        return "No relevant text passages found.", []

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

    learning_mode = st.session_state.get("learning_mode", "Standard Chat")

    chain = get_rag_chain(llm, learning_mode, response_lang)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    with st.spinner("Antwort wird generiert..."):
        try:
            answer = chain.invoke({
                "chat_history": chat_history,
                "context": context_text,
                "question": user_question
            })
            sources = format_sources(docs)
            if debug_mode:
                st.info(f"[DEBUG] Antwortsprache: {response_language_name}")
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
                with st.expander("Gefundene Quellen (Folien)"):
                    for src in sources:
                        st.markdown(f"- **{src['label']}**")
                        st.caption(f"_{src['snippet']}_")


def main():
    st.title("Lern-RAG für Vorlesungen")
    st.write("Persistente Wissensdatenbank nach Fächern / Kursen.")
    st.divider()

    course_name, groq_api_key, persist_index = render_sidebar()

    if not groq_api_key:
        st.warning(
            "Kein API Key gefunden. Lege GROQ_API_KEY in .env ab oder gib ihn links ein.")
        return

    st.subheader(f"Aktuelles Fach: {course_name}")

    uploaded_files = st.file_uploader(
        f"Lade neue PDF-Folien für '{course_name}' hoch (optional)",
        type="pdf",
        accept_multiple_files=True,
    )

    safe_course_name = "".join(
        [c for c in course_name if c.isalnum() or c in (' ', '-', '_')]).strip()
    if not safe_course_name:
        safe_course_name = "default_course"
    has_index = (INDEX_ROOT / safe_course_name / "index.faiss").exists()

    if not uploaded_files and not has_index:
        st.info(
            f"Bitte lade mindestens ein PDF hoch, um das Fach '{course_name}' zu starten.")
        return

    embeddings = get_embeddings_model()

    try:
        with st.spinner("Index wird geladen / aktualisiert..."):
            vector_store, course_idx_key, stats, loaded_from_disk = update_or_load_vector_store(
                course_name,
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

    if st.session_state.index_key != course_idx_key:
        # Chat zurücksetzen, wenn das Fach gewechselt wird
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": f"Wissensdatenbank für '{course_name}' geladen. Was möchtest du lernen?",
                "sources": [],
            }
        ]

    st.session_state.vector_store = vector_store
    st.session_state.index_key = course_idx_key
    st.session_state.doc_stats = stats

    if loaded_from_disk:
        if uploaded_files:
            st.caption(
                "Index existierte, aber keine neuen/unbekannten Dateien gefunden.")
        else:
            st.caption("Index aus lokaler Speicherung geladen.")
    else:
        st.caption("Index wurde mit neuen Dateien aktualisiert.")

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
