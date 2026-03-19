# 📚 AI Research Paper RAG Assistant

---

## 📌 Project Overview & Problem Statement

When preparing submissions for academic conferences, conducting a comprehensive literature review is often a massive bottleneck. Researchers spend hours filtering through rigid keyword searches to find relevant prior work, which is highly inefficient given the sheer volume of daily AI/ML publications.

This project solves that problem by providing an intelligent, semantic search engine and Retrieval-Augmented Generation (RAG) assistant specifically tailored for AI/ML researchers. By leveraging vector embeddings, the system understands the *context* and *meaning* of a user's query, retrieves the most highly relevant research papers, and constructs a precise context window for an LLM to synthesize a grounded answer.

> **Note:** This project is built directly on a forked version of the official Endee repository, in strict compliance with the assessment guidelines.

---

## 🏗️ System Design & Technical Approach

The architecture follows a modern RAG pipeline, decoupled into an automated ingestion script and an interactive retrieval interface:

1. **Data Ingestion**
   A Python script dynamically fetches metadata and abstracts for 100 recent AI/ML/NLP papers directly from ArXiv.

2. **Embedding Generation**
   The `sentence-transformers` library (`all-MiniLM-L6-v2`) converts the text abstracts into 384-dimensional dense vector representations.

3. **Vector Database**
   **Endee** acts as the core storage and retrieval engine, holding the vectors and paper metadata.

4. **User Interface**
   A Streamlit application provides a clean, local web interface.

5. **Retrieval & Prompting**
   User queries are embedded in real-time, matched against the Endee database using cosine similarity, and structured into a strict prompt template ready for an LLM API to generate a final answer.

---

## 🧠 Explanation of How Endee is Used

Endee is the foundational vector database for this application, replacing traditional keyword-based SQL/NoSQL databases to enable true semantic search.

* **Initialization**
  The app connects to a locally hosted Endee instance via its REST API and initializes an index (`research_papers`) configured for INT8 precision and cosine similarity.

* **Storage (Upsertion)**
  During ingestion, high-dimensional vectors (generated from paper abstracts) are batched and **upserted** into Endee alongside their corresponding metadata (Title, Authors, Abstract, URL).

* **Semantic Search (Querying)**
  When a user asks a question, the Streamlit app passes the embedded query vector to Endee's `query()` endpoint. Endee rapidly performs an Approximate Nearest Neighbor (ANN) search to return the top *K* most semantically relevant papers to build the RAG context.

---

## ⚙️ Setup and Execution Instructions

### 📋 Prerequisites

* Docker Desktop *(required to run the Endee server)*
* Python 3.9+

---

### 🚀 Step 1: Start the Endee Vector Database

Open your terminal and run:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

> **Windows Users:** If using Command Prompt, replace `\` with `^`.

The database will be available at:
👉 http://localhost:8080

---

### 📦 Step 2: Install Python Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 📥 Step 3: Run the Ingestion Pipeline

```bash
python ingest.py
```

This will populate your Endee database with recent AI research papers from ArXiv.

---

### 🖥️ Step 4: Launch the Streamlit App

```bash
streamlit run app.py
```

---

## ✅ Final Notes

* Ensure the Endee server is running before launching the app
* The system performs real-time semantic search using vector embeddings
* Designed to demonstrate practical usage of vector databases and RAG pipelines

---
