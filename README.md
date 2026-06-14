# ⚖️ LegalBot: A Retrieval-Augmented Generation System for Legal Assistance and Interpretation of Domestic Violence Cases

## Overview

LegalBot is an AI-powered legal assistance system designed to support the analysis of domestic violence and dowry-related legal cases. The system leverages a **Retrieval-Augmented Generation (RAG)** architecture to enable legal professionals to upload case files and obtain context-aware answers to domain-specific legal queries.

Unlike conventional legal chatbots that provide generic legal information, LegalBot performs **case-specific interpretation** by combining semantic retrieval from annotated legal knowledge with large language model generation.

---

## Key Features

* 📄 Upload and analyze new legal case files
* 🔍 Retrieval-Augmented Generation (RAG) pipeline
* ⚖️ Specialized for Domestic Violence and Dowry Death cases
* 🧠 Semantic search using Sentence-BERT embeddings
* 💬 Natural language legal question answering
* 📚 Evidence-backed responses with retrieved context
* 📊 Relevance scoring using cosine similarity
* 💰 Fully open-source and cost-effective architecture

---

## System Architecture

### Phase 1: Knowledge Base Creation

* Raw domestic violence case files converted to text format
* Manual annotation of legal parameters including:

  * Facts of Case
  * Trial Court Judgment
  * Dowry
  * Evidence
  * Witness Statements
  * IPC/CrPC Sections
  * Outcome
  * Final Judgment
* Annotated data stored in Excel format
* Excel annotations converted into a structured JSON knowledge base

### Phase 2: Retrieval Layer

* Sentence-BERT (`all-MiniLM-L6-v2`) generates vector embeddings
* User queries are converted into embeddings
* Cosine similarity identifies the most relevant legal contexts
* Top-ranked contexts are retrieved for grounding

### Phase 3: Generation Layer

* Retrieved contexts combined with uploaded case content
* Prompt constructed dynamically
* FLAN-T5 generates concise legal responses
* Output displayed with supporting evidence and relevance score

---

## Tech Stack

| Component        | Technology                          |
| ---------------- | ----------------------------------- |
| Frontend         | Streamlit                           |
| Embedding Model  | Sentence-BERT (all-MiniLM-L6-v2)    |
| LLM              | Google FLAN-T5                      |
| Retrieval Method | Cosine Similarity                   |
| Data Storage     | JSON Knowledge Base                 |
| Data Annotation  | Excel                               |
| Language         | Python                              |
| NLP Libraries    | Transformers, Sentence Transformers |
| Deployment       | Streamlit Local Deployment          |

---

## Dataset

The system was developed using a manually annotated corpus of **10 real-world domestic violence case files**.

### Annotation Labels

* Trial Court Judgment
* Facts of Case
* Sections
* Victim Death
* Dowry
* Evidence
* PW Statement
* IPC / CrPC
* Fact Map with IPC
* Outcome
* Final Judgment

The annotations were used to construct a domain-specific legal knowledge base for semantic retrieval.

---

## Project Structure

```text
LegalBot/
│
├── app.py
├── rag_utils.py
├── convert_to_json.py
├── TrainingDataFINAL.xlsx
├── QuestionMapping.xlsx
├── legal_knowledge.json
├── requirements.txt
├── README.md
│
└── case_files/
    ├── case1.txt
    ├── case2.txt
    └── ...
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/LegalBot.git
cd LegalBot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Sample Queries

Examples of lawyer-oriented questions supported by the system:

* Is there any demand for dowry?
* Was the death caused by abnormal circumstances?
* How many prosecution witnesses are hostile?
* Was there any declaration by the victim?
* Did the cruelty occur soon before death?
* Does the prosecution witness statement match the complainant's statement?

---

## Research Contributions

* Developed a case-file-aware legal assistant capable of analyzing uploaded legal documents.
* Introduced a domain-specific RAG pipeline for domestic violence and dowry-related cases.
* Built a legal professional–oriented query system rather than a public legal information chatbot.
* Demonstrated a cost-effective legal AI solution using fully open-source technologies.
* Improved transparency through evidence-backed responses and relevance scoring.

---

## Future Work

* Expansion to larger legal datasets
* Integration with vector databases such as FAISS or ChromaDB
* Support for PDF uploads through OCR pipelines
* Multi-language legal document analysis
* Enhanced explainability and citation-based legal reasoning
* Fine-tuned legal language models for Indian jurisprudence

---

## Disclaimer

This project is intended for research and educational purposes only. The generated responses should not be considered legal advice. Legal professionals must independently verify all outputs before making legal decisions.

---

## Authors

Developed as part of a research project on the application of Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) for legal assistance in domestic violence case analysis.
