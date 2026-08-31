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
                │
                ▼
     [ Groq / Gemini LPU Engine ]
                │
                ▼
     [ Real-Time Streamed Output ]
📋 Comparative AnalysisDimensionStandard Campus ChatbotsAmrapali AI AssistantResponse Latency4–7 seconds batch waitInstant sub-second token streamingContext GroundingGeneric public LLM memoryChromaDB vector retrieval (Strict RAG)State Quota IntelligenceIgnores domicile statusAutomates 25% Uttarakhand concessionInterface ControlsSingle plain prompt inputAction chips, session reset, export toolCredential SecurityRisk of hardcoded tokensEncrypted Cloud Secrets management🗂️ Project StructurePlaintextamrapali-assistant/
├── .streamlit/
│   └── secrets.toml          # Local environment API keys (git-ignored)
├── amrapali_data.txt         # University prospectus, courses, fees, hostels
├── requirements.txt          # Production dependencies
├── streamlit_app.py          # Main application & interactive UI
└── README.md                 # Project architecture & documentation
⚙️ Local Setup & Deployment1. Clone RepositoryBashgit clone [https://github.com/mohit-manik-bhatt-1/amrapali-assistant.git](https://github.com/mohit-manik-bhatt-1/amrapali-assistant.git)
cd amrapali-assistant
2. Configure EnvironmentBashpython -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
3. Configure CredentialsCreate .streamlit/secrets.toml and add your inference key:Ini, TOMLGROQ_API_KEY = "gsk_..."
4. Launch Local InstanceBashstreamlit run streamlit_app.py
📚 Knowledge Domain CoverageFaculty of Technology & Computer Applications: MCA, BCA, B.Tech (CSE, AI/ML, Data Science).Faculty of Commerce & Business Management: MBA (Dual Specialization), BBA, B.Com (Hons).Faculty of Hospitality Management: BHM, BHMCT, Professional Culinary Diplomas.Faculty of Pharmacy & Education: B.Pharm, D.Pharm, B.Ed.Campus Life & Logistics: 19-acre infrastructure, hostel allotment rules, mess plans, and CTPD placement records.👨‍💻 Author & MaintainerDeveloper: Mohit Manik BhattDepartment: Master of Computer Applications (MCA)Institution: Amrapali University, Haldwani
