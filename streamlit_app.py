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

# UI Header
st.markdown(
    """
    <div style='text-align: center; padding: 18px; background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7); color: white; border-radius: 12px; margin-bottom: 20px;'>
        <h2 style='margin:0; font-size: 24px;'>Amrapali University AI Assistant</h2>
        <p style='margin:5px 0 0 0; opacity: 0.9;'>Official Campus Digital Helpdesk • Running on Gemini 3.6 Flash</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 1. API Client Setup
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is missing. Add it to Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Knowledge Base & Vector Store Setup
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

# 3. Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Amrapali University AI Assistant. How can I help you today with admissions, courses, fees, or campus facilities?"}
    ]

# Preset Buttons
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

# 4. Handle Input
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
            "Answer politely and professionally using ONLY the provided context below. "
            "Use clear bullet points and bold keys. "
            "If details cannot be found in context, instruct the user to contact admission@amrapali.ac.in.\n\n"
            f"Context Data:\n{context_text}"
        )

        with st.spinner("Thinking..."):
            try:
                # Interactions API invocation (standard for Gemini 3.6 Flash)
                interaction = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=user_query,
                    system_instruction=system_instruction
                )
                reply = interaction.outputs[-1].text
            except Exception as e1:
                # Fallback to generate_content if interactions endpoint differs on your environment
                try:
                    res = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=user_query,
                        config={"system_instruction": system_instruction}
                    )
                    reply = res.text
                except Exception as e2:
                    reply = f"API Error: {str(e2)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
