import unittest

from models import DocumentStats
from rag_engine import (
    chunk_texts_with_page_metadata,
    infer_response_language,
    sanitize_course_name,
)
from rag_service import build_chat_history, format_sources


class FakeDoc:
    def __init__(self, source: str, page: int, content: str):
        self.metadata = {"source": source, "page": page}
        self.page_content = content


class RagCoreTests(unittest.TestCase):
    def test_sanitize_course_name_returns_default_for_invalid_name(self):
        self.assertEqual(sanitize_course_name("***"), "default_course")

    def test_infer_response_language_detects_german(self):
        self.assertEqual(infer_response_language(
            "Bitte erklaere die Frage aus dem Dokument"), "de")

    def test_build_chat_history_uses_last_messages(self):
        messages = [
            {"role": "assistant", "content": "Hallo"},
            {"role": "user", "content": "Frage 1"},
            {"role": "assistant", "content": "Antwort 1"},
            {"role": "user", "content": "Frage 2"},
        ]
        history = build_chat_history(messages, max_messages=2)
        self.assertEqual(history, "Assistant: Antwort 1\nUser: Frage 2")

    def test_format_sources_deduplicates_by_source_and_page(self):
        docs = [
            FakeDoc("Skript.pdf", 1, "A " * 50),
            FakeDoc("Skript.pdf", 1, "B " * 50),
            FakeDoc("Skript.pdf", 2, "C " * 50),
        ]

        sources = format_sources(docs)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].label, "Skript - Folie 1")
        self.assertTrue(len(sources[0].snippet) <= 283)

    def test_document_stats_roundtrip(self):
        payload = {
            "word_count": 10,
            "chunk_count": 2,
            "file_count": 1,
            "page_count": 3,
            "processed_files": ["a.pdf"],
        }
        stats = DocumentStats.from_mapping(payload)
        self.assertEqual(stats.to_mapping(), payload)

    def test_chunk_texts_with_page_metadata_preserves_page_numbers(self):
        class FakeSplitter:
            def split_text(self, text: str):
                mapping = {
                    "Seite 1": ["Chunk 1A", "Chunk 1B"],
                    "Seite 2": ["Chunk 2A"],
                }
                return mapping.get(text, [])

        page_texts = [(1, "Seite 1"), (2, "Seite 2")]
        texts, metadatas = chunk_texts_with_page_metadata(
            page_texts=page_texts,
            splitter=FakeSplitter(),
            file_name="Skript.pdf",
        )

        self.assertEqual(texts, ["Chunk 1A", "Chunk 1B", "Chunk 2A"])
        self.assertEqual([m["page"] for m in metadatas], [1, 1, 2])
        self.assertEqual([m["chunk"] for m in metadatas], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
