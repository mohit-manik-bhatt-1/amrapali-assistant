import streamlit as st
import os
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(
    page_title="Amrapali University Smart AI",
    page_icon="🎓",
    layout="centered"
)

# 1. API Client Setup
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is missing. Add it to Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Automatically Detect Supported Model
@st.cache_resource
def get_supported_model():
    candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    try:
        available = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
            if any("generateContent" in act for act in actions):
                available.append(m.name.replace("models/", ""))
        for cand in candidates:
            if cand in available:
                return cand
        return available[0] if available else "gemini-2.5-flash"
    except Exception:
        return "gemini-2.5-flash"

ACTIVE_MODEL = get_supported_model()

# 3. Custom UI Header
st.markdown(
    f"""
    <div style='text-align: center; padding: 18px; background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7); color: white; border-radius: 12px; margin-bottom: 20px;'>
        <h2 style='margin:0; font-size: 24px;'>Amrapali University AI Assistant</h2>
        <p style='margin:5px 0 0 0; opacity: 0.9;'>Official Campus Digital Helpdesk • Active Model: {ACTIVE_MODEL}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 4. Knowledge Base Setup
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. User Query Handling
user_query = st.chat_input("Ask about Amrapali University...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        matched_docs = retriever.invoke(user_query)
        context_text = "\n\n".join(d.page_content for d in matched_docs)

        system_instruction = (
            "You are the official Amrapali University AI Assistant. "
            "Answer politely and professionally using ONLY the provided context. "
            "Use clear bullet points and bold keys. "
            "If details cannot be found, advise contacting admission@amrapali.ac.in.\n\n"
            f"Context Data:\n{context_text}"
        )

        def stream_response():
            try:
                response_stream = client.models.generate_content_stream(
                    model=ACTIVE_MODEL,
                    contents=user_query,
                    config={"system_instruction": system_instruction}
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as err:
                yield f"Error calling {ACTIVE_MODEL}: {str(err)}"

        reply = st.write_stream(stream_response)
        st.session_state.messages.append({"role": "assistant", "content": reply})
