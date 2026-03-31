# RAG PDF Chat: AI-Lernassistent für die Prüfungsvorbereitung (Streamlit)

Dieses Projekt ist eine Retrieval-Augmented Generation (RAG) Anwendung, die speziell für die strukturierte Aufbereitung von Vorlesungsfolien und Studienskripten entwickelt wurde. Die App liest hochgeladene PDFs aus, erstellt Embeddings und speichert diese fachbasiert in einem persistenten FAISS-Index ab. Fragen werden über ein Groq-Chatmodell durch den Kontext der eigenen Skripte beantwortet.

Aktueller Modell-Default in der App:
- `llama-3.3-70b-versatile`

## Was die App kann

- **Fach-basierte Vektor-Indizes:** Dokumente können in logischen "Kursen" oder "Fächern" gespeichert werden.
- **Persistenz & Inkrementelle Updates:** Der Index wird lokal auf der Festplatte gespeichert. Neue PDF-Folien können zu einem existierenden Kurs hinzugefügt werden, ohne dass alte Folien neu eingebettet werden müssen.
- **Mehrere Lern-Modi (Prompting):**
  - *Standard Chat:* Beantwortet direkte Fragen zu den Folien.
  - *Quiz-Master:* Die KI generiert Klausurfragen aus dem Text und bewertet deine Antworten.
  - *Sokratisch:* Die KI fasst Konzepte zusammen und stellt Gegenfragen, um das Verständnis zu prüfen.
- **Saubere Quellenanzeige:** Chat-Antworten werden mit deduplizierten Referenzen belegt (z. B. `📄 Vorlesung_03_SQL – Folie 42`), damit Antworten auf Basis des echten Skripts nachvollziehbar bleiben.
- **RAG-Tuning im UI:** Basis-Parameter (Chunk Size, Overlap, Anzahl abgerufener Chunks (k), Temperature) lassen sich on-the-fly im Frontend anpassen.

## Was die App nicht kann

- Kein OCR für gescannte PDFs, die nur aus Bildern ohne echten Text-Layer bestehen.
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

*(Die App kann den Key alternativ auch bei jeder Sitzung über das Sidebar-Inputfeld in der UI entgegennehmen.)*

## Start

```bash
streamlit run app.py
```

Standard-URL lokal: `http://localhost:8501`

## Technische Kurzbeschreibung

Die App ist modular aufgebaut, wobei das Frontend (`app.py`) von der Backend-Logik (`rag_engine.py`) getrennt ist.
- **UI:** Streamlit (`app.py`)
- **Backend/RAG-Controller:** Python (`rag_engine.py`)
- **PDF Parsing:** PyPDF2
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS (mit lokaler Serialisierung)
- **LLM / Orchestrierung:** ChatGroq via LangChain

## Hinweise zur Nutzung für die Prüfungsvorbereitung
- Wenn das hochgeladene PDF sehr grafisch ist, werden ggf. weniger Chunks extrahiert.
- Der **Quiz-Master Modus** eignet sich ideal für Reproduktionsübungen. Formuliere auf die gestellten Fragen ruhig aus, die KI bewertet den Input sehr präzise anhand der Skripte.
- Indizes werden im lokalen Verzeichnis `.indexes/` abgelegt. Um einen Kurs komplett zurückzusetzen, kann der betroffene Kursordner dort einfach gelöscht werden.

## Lizenz

MIT
