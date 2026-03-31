import unittest

from models import DocumentStats
from rag_engine import infer_response_language, sanitize_course_name
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


if __name__ == "__main__":
    unittest.main()
