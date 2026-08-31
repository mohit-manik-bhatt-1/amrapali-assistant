import streamlit as st
import os
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Amrapali University AI Assistant", page_icon="🎓", layout="centered")

st.markdown(
    """
    <div style='text-align: center; padding: 18px; background: linear-gradient(135deg, #1e3a8a, #0284c7); color: white; border-radius: 12px; margin-bottom: 25px;'>
        <h2 style='margin:0; font-size: 24px;'>Amrapali University AI Assistant</h2>
        <p style='margin:5px 0 0 0; opacity: 0.9;'>Official Campus Digital Helpdesk</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Read encrypted secret key from Streamlit Cloud
api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_retriever():
    if os.path.exists("amrapali_data.txt"):
        loader = TextLoader("amrapali_data.txt")
        docs = loader.load()
    else:
        docs = [Document(page_content="Amrapali University Haldwani. MCA Fee: ₹63,000. Hostel: ₹72,000.")]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about MCA, fees, syllabus, or hostel facilities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        matched_docs = retriever.invoke(prompt)
        context_text = "\n\n".join(d.page_content for d in matched_docs)

        system_instruction = (
            "You are the official Amrapali University Digital Assistant. "
            "Strictly answer based ONLY on the context below. Format output clearly with bullet points. "
            "If the answer is not present in the context, instruct the user to email admission@amrapali.ac.in.\n\n"
            f"Context:\n{context_text}"
        )

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"system_instruction": system_instruction}
            )
            reply = response.text
        except Exception as e:
            reply = f"Error generating response: {str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
