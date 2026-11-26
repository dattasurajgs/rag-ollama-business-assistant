# 📚 RAG Business Assistant (Ollama + ChromaDB)

A beginner-friendly, locally running document Q&A chatbot built with **Streamlit**, **Ollama**, and **ChromaDB**. Upload PDF or HTML documents, then ask questions grounded in their contents.

## Features

- PDF and HTML text extraction
- Overlapping text chunking
- Local embeddings through Ollama's `all-minilm` model
- Persistent vector storage with ChromaDB
- Context-grounded answers using TinyLlama
- Source filename and page references
- Multiple-question chat history
- Local document processing (your files are not sent to a hosted AI API)

> Scanned/image-only PDFs may not contain extractable text. OCR is not included.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running

## Run on Windows

Open PowerShell in the project folder.

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the local models

Make sure Ollama is running, then execute:

```powershell
ollama pull tinyllama
ollama pull all-minilm
```

### 4. Start the chatbot

```powershell
streamlit run app.py
```

Open the local address Streamlit displays, usually http://localhost:8501.

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull tinyllama
ollama pull all-minilm
streamlit run app.py
```

Ensure the Ollama service is running before starting the app.

## How it works

1. Upload a PDF or HTML document.
2. Extract text and split it into overlapping chunks.
3. Ollama generates an embedding for each chunk.
4. ChromaDB stores embeddings and metadata in `chroma_db/`.
5. A question is embedded and matched against relevant chunks.
6. TinyLlama generates an answer from the retrieved context with source references.

## Project structure

```text
rag-ollama-business-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/          # Created automatically; not committed
```

## Troubleshooting

- **Cannot connect to Ollama:** Open the Ollama app or start its service, then retry.
- **Model not found:** Pull both models using the commands above.
- **No readable text:** The PDF may be scanned. OCR is not included; use a text-searchable PDF.
- **Port already in use:** Run `streamlit run app.py --server.port 8502`.
- **Reset indexed documents:** Stop the app and remove `chroma_db/`. This deletes the local index.

## Notes

- Change `LLM_MODEL` in `app.py` to use another model already installed in Ollama.
- Keep the embedding model consistent with the stored vectors. If you change `EMBED_MODEL`, clear and rebuild the index.
- This is a learning/demo project. Independently verify important business, financial, or legal information.

## Author

Created by [dattasurajgs](https://github.com/dattasurajgs).
