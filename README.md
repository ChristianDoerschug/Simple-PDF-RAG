# RAG PDF Chat: AI-Lernassistent für die Prüfungsvorbereitung (Streamlit)

Dieses Projekt ist eine Retrieval-Augmented Generation (RAG) Anwendung, die speziell für die strukturierte Aufbereitung von Vorlesungsfolien und Studienskripten entwickelt wurde. Die App liest hochgeladene PDFs aus, erstellt Embeddings und speichert diese fachbasiert in einem persistenten FAISS-Index ab. Fragen werden über ein Groq-Chatmodell durch den Kontext der eigenen Skripte beantwortet.

Aktueller Modell-Default in der App:

- `llama-3.3-70b-versatile`

## Was die App kann

- **Fach-basierte Vektor-Indizes:** Dokumente können in logischen "Kursen" oder "Fächern" gespeichert werden.
- **Persistenz & Inkrementelle Updates:** Der Index wird lokal auf der Festplatte gespeichert. Neue PDF-Folien können zu einem existierenden Kurs hinzugefügt werden, ohne dass alte Folien neu eingebettet werden müssen.
- **OCR für gescannte PDFs:** PDF-Import läuft über Docling und kann dadurch auch gescannte Dokumente (Bild-PDFs) verarbeiten.
- **Semantisches Chunking:** Inhalte werden mit dem LangChain `SemanticChunker` (aus `langchain-experimental`) semantisch segmentiert; falls das Paket nicht verfügbar ist, wird auf `RecursiveCharacterTextSplitter` zurückgefallen.
- **Umschaltbare OCR-Engine:** In der Sidebar kann zwischen `easyocr` und `rapidocr` gewechselt werden.
- **Mehrere Lern-Modi (Prompting):**
  - _Standard Chat:_ Beantwortet direkte Fragen zu den Folien.
  - _Quiz-Master:_ Die KI generiert Klausurfragen aus dem Text und bewertet deine Antworten.
  - _Sokratisch:_ Die KI fasst Konzepte zusammen und stellt Gegenfragen, um das Verständnis zu prüfen.
- **Saubere Quellenanzeige:** Chat-Antworten werden mit deduplizierten Referenzen belegt (z. B. `📄 Vorlesung_03_SQL – Folie 42`), damit Antworten auf Basis des echten Skripts nachvollziehbar bleiben.
- **RAG-Tuning im UI:** Parameter wie Anzahl abgerufener Chunks (k), Temperature und OCR-Engine lassen sich on-the-fly im Frontend anpassen.

## Was die App nicht kann

- Keine Benutzerverwaltung oder Multi-User-Isolation (Lokal für Einzelnutzer konzipiert).

## Voraussetzungen

- Python 3.11+
- Groq API Key

## Installation

1. Virtuelle Umgebung erstellen:

```bash
python -m venv .venv
```

2. Umgebung aktivieren (Windows PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

3. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## API Key konfigurieren

Lege im Projektordner eine Datei `.env` an:

```env
GROQ_API_KEY=dein_key
```

_(Die App kann den Key alternativ auch bei jeder Sitzung über das Sidebar-Inputfeld in der UI entgegennehmen.)_

## Start

```bash
streamlit run app.py
```

Standard-URL lokal: `http://localhost:8501`

## Tests

Es gibt eine kleine Unit-Test-Suite für zentrale Kernfunktionen.

Ausführen (im aktivierten venv):

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Technische Kurzbeschreibung

Die App ist modular aufgebaut und trennt UI, RAG-Engine, Service-Orchestrierung und Datamodels.

- **UI:** Streamlit (`app.py`)
- **RAG-Engine:** Indexing, PDF-Chunking, Prompt-Builder (`rag_engine.py`)
- **Service-Schicht:** Chatverlauf-Aufbereitung, Quellenformatierung, Pipeline-Orchestrierung (`rag_service.py`)
- **Datamodels:** Strukturierte Objekte für Stats, Quellen und Pipeline-Ergebnisse (`models.py`)
- **Tests:** Unit-Tests für Kernlogik (`tests/test_rag_core.py`)
- **PDF Parsing & OCR:** Docling (`DocumentConverter` mit `DocumentStream`)
- **Chunking:** Primär `SemanticChunker` (`langchain_experimental.text_splitter`), Fallback `RecursiveCharacterTextSplitter`
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS (mit lokaler Serialisierung)
- **LLM / Orchestrierung:** ChatGroq via LangChain

## Hinweise zur Nutzung für die Prüfungsvorbereitung

- Wenn das hochgeladene PDF sehr grafisch ist, werden ggf. weniger Chunks extrahiert.
- Für gescannte PDFs auf CPU ist `rapidocr` oft schneller; `easyocr` ist in vielen Fällen robuster bei schwierigen Layouts.
- Der **Quiz-Master Modus** eignet sich ideal für Reproduktionsübungen. Formuliere auf die gestellten Fragen ruhig aus, die KI bewertet den Input sehr präzise anhand der Skripte.
- Indizes werden im lokalen Verzeichnis `.indexes/` abgelegt. Um einen Kurs komplett zurückzusetzen, kann der betroffene Kursordner dort einfach gelöscht werden.

## Performance- und Warnhinweise

- Beim ersten OCR-Lauf lädt Docling Modelle nach; der erste Import kann daher spürbar länger dauern.
- Hinweis wie `Xet Storage is enabled ... hf_xet not installed` ist nicht kritisch, aber für schnellere Downloads sinnvoll:

```bash
pip install hf_xet
```

- Die Torch-Meldung zu `pin_memory` auf CPU ist ein Hinweis, kein Fehler. Funktional läuft die Verarbeitung trotzdem korrekt auf CPU.

## Lizenz

MIT
