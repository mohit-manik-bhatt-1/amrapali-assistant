import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Amrapali University AI Assistant", page_icon="🎓", layout="centered")

# 1. Groq Client Authentication
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# 2. Auto-Detect Supported Groq Model
@st.cache_resource
def get_working_groq_model():
    try:
        models_data = client.models.list()
        active_ids = [m.id for m in models_data.data]
        
        # Priority check for fast conversational models
        preference = [
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "compound-mini"
        ]
        for candidate in preference:
            if candidate in active_ids:
                return candidate
        
        # Fallback to the first available text generation model
        for m_id in active_ids:
            if not any(x in m_id for x in ["whisper", "guard", "orpheus"]):
                return m_id
        return active_ids[0]
    except Exception as e:
        # Safe fallback
        return "openai/gpt-oss-20b"

ACTIVE_MODEL = get_working_groq_model()

# 3. UI Header
st.markdown(
    f"""
    <div style='text-align: center; padding: 18px; background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7); color: white; border-radius: 12px; margin-bottom: 20px;'>
        <h2 style='margin:0; font-size: 24px;'>Amrapali University AI Assistant</h2>
        <p style='margin:5px 0 0 0; opacity: 0.9;'>Official Campus Digital Helpdesk • Engine: {ACTIVE_MODEL}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 4. Vector DB Setup (Amrapali University Knowledge Base)
@st.cache_resource
def load_knowledge_base():
    if os.path.exists("amrapali_data.txt"):
        loader = TextLoader("amrapali_data.txt")
        docs = loader.load()
    else:
        docs = [Document(page_content="Amrapali University Haldwani. MCA Fee: ₹63,000. Hostel: ₹72,000.")]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_knowledge_base()

# 5. Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Amrapali University AI Assistant. How can I help you today with admissions, courses, fees, or campus facilities?"}
    ]

# Action Chips
col1, col2, col3, col4 = st.columns(4)
suggested_prompt = None
with col1:
    if st.button("💰 MCA & BTech Fees"):
        suggested_prompt = "What is the fee structure for MCA and B.Tech?"
with col2:
    if st.button("🏢 Campus & Labs"):
        suggested_prompt = "Tell me about the university infrastructure, computer labs, and library."
with col3:
    if st.button("🛏️ Hostel Facilities"):
        suggested_prompt = "What are the hostel charges, room types, and mess facilities?"
with col4:
    if st.button("💼 Placements"):
        suggested_prompt = "What are the placement statistics and top recruiting companies?"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about Amrapali University...") or suggested_prompt

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        matched_docs = retriever.invoke(user_query)
        context_text = "\n\n".join(d.page_content for d in matched_docs)

        system_instruction = (
            "You are the official Amrapali University AI Assistant. "
            "Answer politely, accurately, and professionally using ONLY the provided context below. "
            "Use clear bullet points and bold keys. "
            "If details cannot be found in context, instruct the user to contact admission@amrapali.ac.in.\n\n"
            f"Context Data:\n{context_text}"
        )

        def stream_groq():
            try:
                completion = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query}
                    ],
                    stream=True
                )
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield f"Inference Error: {str(e)}"

        reply = st.write_stream(stream_groq)
        st.session_state.messages.append({"role": "assistant", "content": reply})
