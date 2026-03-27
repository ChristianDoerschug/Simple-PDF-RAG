# RAG PDF Chat (Streamlit)

Dieses Projekt ist eine kleine RAG-Anwendung fuer PDF-Dateien.
Sie liest ein hochgeladenes PDF, erstellt Embeddings, speichert diese in einem FAISS-Index und beantwortet Fragen mit einem Groq-Chatmodell.

Aktueller Modell-Default in der App:
- llama-3.3-70b-versatile

## Was die App kann

- Ein PDF hochladen und den Text extrahieren
- Text in Chunks aufteilen und lokal embeddieren
- Relevante Chunks per Aehnlichkeitssuche abrufen
- Fragen im Chat stellen, inklusive Follow-up Fragen im gleichen Verlauf
- Basis-Parameter in der Sidebar anpassen (Chunk Size, Overlap, k, Temperature)

## Was die App nicht kann

- Kein OCR fuer gescannte PDFs ohne echten Textlayer
- Keine Quellenzitate mit Seitenzahl im UI
- Kein persistenter Index auf Platte (Index lebt nur waehrend der Session)
- Keine Benutzerverwaltung oder Multi-User-Isolation

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

3. Abhaengigkeiten installieren:

```bash
pip install -r requirements.txt
```

## API Key konfigurieren

Lege im Projektordner eine Datei .env an:

```env
GROQ_API_KEY=dein_key
```

Die App kann den Key alternativ auch aus dem Sidebar-Input nehmen.

## Start

```bash
streamlit run app.py
```

Standard-URL lokal: http://localhost:8501

## Technische Kurzbeschreibung

- UI: Streamlit
- PDF Parsing: PyPDF2
- Chunking: RecursiveCharacterTextSplitter
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS
- LLM: ChatGroq (llama-3.3-70b-versatile)

## Bekannte Fehlerbilder

- Leeres oder gescanntes PDF: Es werden keine Chunks erzeugt
- Ungueltiger oder fehlender API Key: Modellaufruf schlaegt fehl
- Modell-Deprecation bei Providerwechsel: Modellnamen in der App anpassen

## Hinweise zur Nutzung

- Antworten koennen unvollstaendig oder falsch sein, wenn relevante Passage nicht im Retrieval landet.
- Bessere Ergebnisse entstehen meist mit praezisen Fragen und sinnvollen k/Chunk-Einstellungen.

## Lizenz

MIT
