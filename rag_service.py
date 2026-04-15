from typing import Any, Callable

from models import RagPipelineResult, SourceReference
from rag_engine import generate_quiz_question, get_rag_chain, infer_response_language


def build_chat_history(messages: list[dict[str, Any]], max_messages: int = 3) -> str:
    recent_messages = messages[-max_messages:]
    convo_lines: list[str] = []
    for message in recent_messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            convo_lines.append(f"User: {content}")
        elif role == "assistant":
            convo_lines.append(f"Assistant: {content}")
    return "\n".join(convo_lines)


def format_sources(docs: list[Any]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[str] = set()

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source", "unbekannt")
        page = meta.get("page", "?")
        doc_id = f"{source}_page_{page}"

        if doc_id in seen:
            continue

        seen.add(doc_id)
        page_content = getattr(doc, "page_content", "")
        snippet = " ".join(page_content.split())
        short_source = str(source).replace(".pdf", "")
        sources.append(
            SourceReference(
                label=f"{short_source} - Folie {page}",
                snippet=snippet[:280] + ("..." if len(snippet) > 280 else ""),
            )
        )

    return sources


def run_rag_pipeline(
    user_question: str,
    vector_store: Any,
    k_retrieval: int,
    learning_mode: str,
    chat_messages: list[dict[str, Any]],
    get_llm_model_fn: Callable[..., Any],
    groq_api_key: str,
    model_name: str,
    temperature: float,
    debug_mode: bool = False,
) -> RagPipelineResult:
    response_lang = infer_response_language(user_question)
    response_language_name = "German" if response_lang == "de" else "English"

    docs = vector_store.similarity_search(user_question, k=k_retrieval)

    if not docs:
        if response_lang == "de":
            return RagPipelineResult(
                answer="Keine relevanten Textstellen gefunden.",
                sources=[],
            )
        return RagPipelineResult(
            answer="No relevant text passages found.",
            sources=[],
        )

    llm = get_llm_model_fn(groq_api_key, model_name, temperature)
    chat_history = build_chat_history(chat_messages)
    chain = get_rag_chain(llm, learning_mode, response_lang)
    # Limit context length to prevent token limit errors
    context_parts = []
    for doc in docs:
        truncated = doc.page_content[:1000] if len(
            doc.page_content) > 1000 else doc.page_content
        context_parts.append(truncated)
    context_text = "\n\n".join(context_parts)

    answer = chain.invoke(
        {
            "chat_history": chat_history,
            "context": context_text,
            "question": user_question,
        }
    )

    debug_messages: list[str] = []
    if debug_mode:
        debug_messages.append(f"Antwortsprache: {response_language_name}")
        debug_messages.append(f"{len(docs)} Dokumente durchsucht")

    return RagPipelineResult(
        answer=answer,
        sources=format_sources(docs),
        debug_messages=debug_messages,
    )


def generate_quiz_from_vector_store(
    vector_store: Any,
    get_llm_model_fn: Callable[..., Any],
    groq_api_key: str,
    model_name: str,
    temperature: float,
    response_lang: str = "de",
    k: int = 3,
) -> tuple[str, list[SourceReference]]:
    random_docs = vector_store.similarity_search(
        "Wichtige Konzepte und Definitionen",
        k=k,
    )
    context_for_question = "\n\n".join(
        [doc.page_content for doc in random_docs])

    llm = get_llm_model_fn(groq_api_key, model_name, temperature)
    quiz_question = generate_quiz_question(
        llm, context_for_question, response_lang)
    return quiz_question, format_sources(random_docs)
