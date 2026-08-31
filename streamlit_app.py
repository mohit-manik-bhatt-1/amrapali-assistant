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

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7);
        padding: 24px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #f8fafc;
    }
    .main-header p {
        font-size: 14px;
        color: #cbd5e1;
        margin: 0;
    }
    .stButton>button {
        border-radius: 20px;
        font-size: 13px;
        padding: 4px 16px;
        border: 1px solid #0284c7;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0284c7;
        color: white;
    }
</style>

<div class="main-header">
    <h1>Amrapali University Smart AI</h1>
    <p>Admissions • Degree Programs • Fee Structure • Hostels • Placements</p>
</div>
""", unsafe_allow_html=True)

# 1. API Client Setup
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is missing. Add it to Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Optimized Vector Store (Cached in memory)
@st.cache_resource
def load_knowledge_base():
    if os.path.exists("amrapali_data.txt"):
        loader = TextLoader("amrapali_data.txt")
        docs = loader.load()
    else:
        docs = [Document(page_content="Amrapali University, Haldwani. Comprehensive campus portal.")]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_knowledge_base()

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the official digital assistant for Amrapali University, Haldwani. How can I help you today with admissions, courses, fees, or campus life?"}
    ]

# 4. Quick Suggestion Chips
col1, col2, col3, col4 = st.columns(4)
suggested_prompt = None

with col1:
    if st.button("💰 MCA & BTech Fees"):
        suggested_prompt = "What is the fee structure for MCA and B.Tech (including UK domicile discount)?"
with col2:
    if st.button("🏢 Campus & Labs"):
        suggested_prompt = "Tell me about the university infrastructure, computer labs, and library."
with col3:
    if st.button("🛏️ Hostel Facilities"):
        suggested_prompt = "What are the hostel charges, room types, and mess facilities?"
with col4:
    if st.button("💼 Placements"):
        suggested_prompt = "What are the placement statistics and top recruiting companies?"

# 5. Display Previous Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Process User Input (Input box or Clicked chip)
user_query = st.chat_input("Ask any question about Amrapali University...") or suggested_prompt

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Retrieve context from ChromaDB
        matched_docs = retriever.invoke(user_query)
        context_text = "\n\n".join(d.page_content for d in matched_docs)

        system_instruction = (
            "You are the official Amrapali University AI Assistant. "
            "Respond politely, professionally, and authoritatively using ONLY the context provided below. "
            "Use clean Markdown formatting, bullet points, and bold keys for scannability. "
            "If asked about UK domicile fees, explicitly highlight the regional discount. "
            "If details cannot be found in the context, advise the user to contact admission@amrapali.ac.in or call campus desk.\n\n"
            f"Context Data:\n{context_text}"
        )

        def stream_response():
            response_stream = client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=user_query,
                config={"system_instruction": system_instruction}
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        reply = st.write_stream(stream_response)
        st.session_state.messages.append({"role": "assistant", "content": reply})
