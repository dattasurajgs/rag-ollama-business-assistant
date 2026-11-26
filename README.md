
# 📚 RAG Business Assistant using Ollama

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG), Ollama, TinyLlama, ChromaDB and Streamlit.

## 🚀 Project Overview

This application allows users to upload PDF and HTML documents and ask questions about their content.

It retrieves relevant information from uploaded documents and uses a local LLM to generate contextual answers.

## 🛠️ Tech Stack

- Python
- Ollama
- TinyLlama
- all-MiniLM Embeddings
- ChromaDB
- Streamlit
- PyPDF
- BeautifulSoup

## 🏗️ Architecture

1. Upload PDF or HTML documents.
2. Extract document text.
3. Split text into smaller chunks.
4. Generate embeddings using all-MiniLM.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks using semantic search.
7. Generate contextual answers using TinyLlama.
8. Display answers through Streamlit.

## 📂 Project Structure

```text
rag-ollama-business-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/  (generated at runtime)
```

## ⚙️ Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download from https://ollama.com/download

### 3. Download models

```bash
ollama pull tinyllama
ollama pull all-minilm
```

### 4. Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 💡 Example Questions

- What are some profitable business ideas?
- What are the low-investment business opportunities?
- Summarize the uploaded document.
- Find online business opportunities.

## 🎯 Key Features

- PDF and HTML document ingestion
- Text chunking
- Semantic embeddings
- Vector database storage
- Context-based question answering
- Local LLM inference

## 👨‍💻 Author

Developed as a hands-on project to explore Retrieval-Augmented Generation and local AI applications.
