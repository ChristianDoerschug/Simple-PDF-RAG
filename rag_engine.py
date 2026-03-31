import io
import hashlib
import json
import logging
from pathlib import Path

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


def get_embeddings_model(model_name: str):
    logger.info("Loading embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def get_llm_model(api_key: str, model_name: str, temperature: float):
    logger.info("Loading LLM model: %s", model_name)
    return ChatGroq(
        temperature=temperature,
        groq_api_key=api_key,
        model_name=model_name,
    )


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


def update_or_load_vector_store(course_name: str, uploaded_files, embeddings, chunk_size: int, chunk_overlap: int, debug_mode: bool = False, persist_index: bool = True):
    safe_course_name = "".join(
        [c for c in course_name if c.isalnum() or c in (' ', '-', '_')]).strip()
    if not safe_course_name:
        safe_course_name = "default_course"

    from pathlib import Path
    index_root = Path(".indexes")
    index_root.mkdir(parents=True, exist_ok=True)

    index_dir = index_root / safe_course_name
    stats_file = index_dir / "stats.json"

    store = None
    stats = {
        "word_count": 0,
        "chunk_count": 0,
        "file_count": 0,
        "page_count": 0,
        "processed_files": []
    }

    if persist_index and (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        if stats_file.exists():
            with stats_file.open("r", encoding="utf-8") as fh:
                loaded_stats = json.load(fh)
                stats.update(loaded_stats)
                if "processed_files" not in stats:
                    stats["processed_files"] = []

    new_files = [
        f for f in uploaded_files if f.name not in stats["processed_files"]]

    if not new_files and store is not None:
        return store, safe_course_name, stats, True

    all_texts = []
    all_metas = []

    for uploaded_file in new_files:
        pdf_data = extract_pdf_chunks(
            uploaded_file.name,
            uploaded_file.getvalue(),
            chunk_size,
            chunk_overlap,
        )
        all_texts.extend(pdf_data["texts"])
        all_metas.extend(pdf_data["metadatas"])
        stats["word_count"] += pdf_data["word_count"]
        stats["page_count"] += pdf_data["page_count"]
        stats["file_count"] += 1
        stats["processed_files"].append(uploaded_file.name)

    if not all_texts and store is None:
        raise ValueError(
            "Keine neuen Chunks erstellt und kein existierender Index gefunden. Bitte lade eine gültige PDF-Datei hoch.")

    if all_texts:
        if store is None:
            store = FAISS.from_texts(
                all_texts, embeddings, metadatas=all_metas)
        else:
            store.add_texts(all_texts, metadatas=all_metas)

        stats["chunk_count"] += len(all_texts)

        if persist_index:
            index_dir.mkdir(parents=True, exist_ok=True)
            store.save_local(str(index_dir))
            with stats_file.open("w", encoding="utf-8") as fh:
                json.dump(stats, fh)

    return store, safe_course_name, stats, False


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


def generate_quiz_question(llm, context_text: str, response_lang: str = "de"):
    """Generate a quiz question from the provided context."""
    response_language_name = "German" if response_lang == "de" else "English"

    prompt = PromptTemplate.from_template(
        """You are a university professor creating exam questions.
Based on the provided document context, generate ONE challenging but fair exam question.
The question should test understanding of key concepts.
Reply strictly in: {response_language}
Do not include any preamble, just the question itself.

Kontext:
{context}

Frage:"""
    )

    chain = (
        {
            "response_language": lambda _: response_language_name,
            "context": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    question = chain.invoke(context_text)
    return question.strip()


def get_rag_chain(llm, learning_mode: str, response_lang: str):
    response_language_name = "German" if response_lang == "de" else "English"

    if learning_mode == "Quiz-Master (Prüfung)":
        system_prompt = """You are a strict but fair university professor testing a student.
Based on the provided document context, generate 1-3 challenging exam questions relating to the student's input or the context.
If the student is answering a question, evaluate their answer based ONLY on the context and explain what they missed.
Do not provide just the direct answer unless asked.
Reply strictly in: {response_language}."""
    elif learning_mode == "Sokratisch (Erklären)":
        system_prompt = """You are a helpful tutor using the Socratic method.
Explain the concepts found in the context based on the student's question clearly and concisely.
However, end your response with a thought-provoking follow-up question to test if the student truly understood the explanation.
Reply strictly in: {response_language}."""
    else:
        system_prompt = """You are a helpful assistant for document Q&A.
Use the document context and chat history to answer follow-up questions.
If the answer is not present in the context, clearly say that the information is not in the document.
Reply strictly in: {response_language}."""

    prompt = PromptTemplate.from_template(
        system_prompt + """

Chatverlauf:
{chat_history}

Kontext:
{context}

Frage:
{question}

Antwort:"""
    )

    return (
        {
            "response_language": lambda _: response_language_name,
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
