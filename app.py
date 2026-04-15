import os
import logging
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from models import DocumentStats
from rag_engine import (
    get_embeddings_model,
    get_llm_model,
    update_or_load_vector_store,
    sanitize_course_name,
)
from rag_service import generate_quiz_from_vector_store, run_rag_pipeline as run_rag_pipeline_service

DEFAULT_OCR_ENGINE = "easyocr"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_K_RETRIEVAL = 2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_ROOT = Path(".indexes")
DEFAULT_LEARNING_MODE = "Standard Chat"
QUIZ_MODE = "Quiz-Master (Prüfung)"
CHAT_BOOT_MESSAGE = "Hi! Lade ein oder mehrere PDFs hoch und stelle Fragen dazu."
CHAT_RESET_MESSAGE = "Chatverlauf wurde geleert. Du kannst direkt weiterfragen."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Multi Input RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def assistant_message(content: str, sources: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "sources": sources or [],
    }


def init_session_state() -> None:
    defaults = {
        "chat_messages": [assistant_message(CHAT_BOOT_MESSAGE)],
        "vector_store": None,
        "index_key": None,
        "debug_mode": False,
        "persist_index": True,
        "doc_stats": DocumentStats().to_mapping(),
        "learning_mode": DEFAULT_LEARNING_MODE,
        "prev_learning_mode": DEFAULT_LEARNING_MODE,
        "quiz_question_generated": False,
        "ocr_engine": DEFAULT_OCR_ENGINE,
        "k_retrieval": DEFAULT_K_RETRIEVAL,
        "temperature": DEFAULT_TEMPERATURE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat(content: str) -> None:
    st.session_state.chat_messages = [assistant_message(content)]


def source_refs_to_dicts(source_refs) -> list[dict[str, str]]:
    return [{"label": f"📄 {source.label}", "snippet": source.snippet} for source in source_refs]


init_session_state()


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


def render_sidebar() -> tuple[str, str, bool]:
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
            [DEFAULT_LEARNING_MODE, QUIZ_MODE, "Sokratisch (Erklären)"],
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
        st.session_state.ocr_engine = st.selectbox(
            "OCR Engine",
            options=["easyocr", "rapidocr"],
            index=0 if st.session_state.ocr_engine == "easyocr" else 1,
            help="EasyOCR ist meist robuster, RapidOCR oft schneller auf CPU.",
        )
        st.session_state.k_retrieval = st.slider(
            "Retrieved Documents (k)",
            min_value=1,
            max_value=5,
            value=DEFAULT_K_RETRIEVAL,
            help="Weniger Dokumente = weniger Tokens, schneller. Max 5 um Rate-Limit zu vermeiden.",
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
            reset_chat(CHAT_RESET_MESSAGE)
            st.rerun()

        return course_name, groq_api_key, st.session_state.persist_index


def render_metrics(stats: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dateien", stats.get("file_count", 0))
    c2.metric("Seiten", stats.get("page_count", 0))
    c3.metric("Woerter", stats.get("word_count", 0))
    c4.metric("Chunks", stats.get("chunk_count", 0))


def run_rag_pipeline(user_question: str, groq_api_key: str, debug_mode: bool = False):
    result = run_rag_pipeline_service(
        user_question=user_question,
        vector_store=st.session_state.vector_store,
        k_retrieval=st.session_state.k_retrieval,
        learning_mode=st.session_state.get(
            "learning_mode", DEFAULT_LEARNING_MODE),
        chat_messages=st.session_state.chat_messages,
        get_llm_model_fn=cached_get_llm_model,
        groq_api_key=groq_api_key,
        model_name=DEFAULT_MODEL,
        temperature=st.session_state.temperature,
        debug_mode=debug_mode,
    )

    if debug_mode:
        for debug_message in result.debug_messages:
            st.info(f"[DEBUG] {debug_message}")

    return result.answer, source_refs_to_dicts(result.sources)


def render_chat() -> None:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander("Gefundene Quellen (Folien)"):
                    for src in sources:
                        st.markdown(f"- **{src['label']}**")
                        st.caption(f"_{src['snippet']}_")


def main() -> None:
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

    safe_course_name = sanitize_course_name(course_name)
    has_index = (INDEX_ROOT / safe_course_name / "index.faiss").exists()

    if not uploaded_files and not has_index:
        st.info(
            f"Bitte lade mindestens ein PDF hoch, um das Fach '{course_name}' zu starten.")
        return

    embeddings = cached_get_embeddings_model()

    try:
        with st.spinner("Index wird geladen / aktualisiert..."):
            vector_store, course_idx_key, stats, loaded_from_disk = update_or_load_vector_store(
                course_name,
                uploaded_files,
                embeddings,
                st.session_state.ocr_engine,
                st.session_state.debug_mode,
                persist_index,
            )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as exc:
        st.error(f"Fehler beim Erstellen/Laden des Index: {exc}")
        return

    if st.session_state.index_key != course_idx_key:
        reset_chat(
            f"Wissensdatenbank für '{course_name}' geladen. Was möchtest du lernen?")

    st.session_state.vector_store = vector_store
    st.session_state.index_key = course_idx_key
    st.session_state.doc_stats = DocumentStats.from_mapping(stats).to_mapping()

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

    if st.session_state.learning_mode == QUIZ_MODE and not st.session_state.quiz_question_generated:
        try:
            with st.spinner("Quiz-Frage wird generiert..."):
                quiz_question, quiz_sources = generate_quiz_from_vector_store(
                    vector_store=st.session_state.vector_store,
                    get_llm_model_fn=cached_get_llm_model,
                    groq_api_key=groq_api_key,
                    model_name=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                    response_lang="de",
                    k=3,
                )

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": f"🎯 **Quiz-Frage:** {quiz_question}",
                        "sources": source_refs_to_dicts(quiz_sources),
                    }
                )
                st.session_state.quiz_question_generated = True
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as exc:
            st.warning(f"Konnte Quiz-Frage nicht generieren: {exc}")

    if st.session_state.learning_mode != QUIZ_MODE:
        st.session_state.quiz_question_generated = False

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
