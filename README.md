# 🎓 Amrapali University AI Campus Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![RAG Architecture](https://img.shields.io/badge/Architecture-RAG%20%2B%20ChromaDB-emerald.svg)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An intelligent, retrieval-augmented campus assistant developed for **Amrapali University, Haldwani**. Delivers real-time answers regarding course curricula, fee concessions, hostel accommodations, and placement statistics with zero hallucination.

---

## ⚡ Key Highlights & USPs

- **Domain-Specific RAG:** Ingests official university documents into an in-memory ChromaDB vector index.
- **Uttarakhand Domicile Awareness:** Pre-configured logic calculating the 25% fee concession for state domicile candidates.
- **Ultra-Fast Token Streaming:** Uses lightweight sentence transformers and token streaming for sub-second responses.
- **Zero-Crash Failover:** Automated fallback mechanisms preventing API rate-limit stalls.
- **Transcript Export:** Export full admission query sessions directly to text/PDF for offline review.

---

## 🏗️ Technical Architecture

```text
  [ Prospective Student / User ]
                │
                ▼
      [ Streamlit Web UI ]
                │
      (Semantic Search Query)
                │
                ▼
     ┌──────────────────────┐
     │  ChromaDB Vector DB  │  ◄── Ingests: amrapali_data.txt
     │ (all-MiniLM-L6-v2)   │
     └──────────┬───────────┘
                │
         Top-K Chunks
                │
                ▼
 ┌──────────────────────────────┐
 │   Prompt Augmentation &      │
 │ Strict Context Verification  │
 └──────────────┬───────────────┘
---

## 📋 Comparative Analysis

| Dimension | Standard Campus Chatbots | Amrapali AI Assistant |
| :--- | :--- | :--- |
| **Response Latency** | 4–7 seconds batch wait | **Instant sub-second token streaming** |
| **Context Grounding** | Generic public LLM memory | **ChromaDB vector retrieval (Strict RAG)** |
| **State Quota Intelligence** | Ignores domicile status | **Automates 25% Uttarakhand concession** |
| **Interface Controls** | Single plain prompt input | **Action chips, session reset, export tool** |
| **Credential Security** | Risk of hardcoded tokens | **Encrypted Cloud Secrets management** |

---

## 🗂️ Project Structure

```text
amrapali-assistant/
├── .streamlit/
│   └── secrets.toml          # Local environment API keys (git-ignored)
├── amrapali_data.txt         # University prospectus, courses, fees, hostels
├── requirements.txt          # Production dependencies
├── streamlit_app.py          # Main application & interactive UI
└── README.md                 # Project architecture & documentation
                │
                ▼
     [ Groq / Gemini LPU Engine ]
                │
                ▼
     [ Real-Time Streamed Output ]
