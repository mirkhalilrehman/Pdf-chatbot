import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load API key from .env
load_dotenv()

st.set_page_config(page_title="📚 RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat with Your PDF")
st.write("Upload a PDF, ask questions, and get context-aware answers!")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("📄 Upload a PDF", type="pdf")

# Session State Initialization
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Process PDF and Create VectorStore
if uploaded_file and not st.session_state.vectorstore:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMENI_API_KEY")
    )
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create memory and chat model
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMENI_API_KEY")
    )

    st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=st.session_state.memory
    )

    st.success("✅ Document processed! You can now ask questions.")

# Show chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Input from user
user_input = st.chat_input("Ask a question about the document...")

# Handle user input
if user_input and st.session_state.rag_chain:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # Get response from chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain.invoke({"question": user_input})
            answer = result["answer"]
            st.markdown(answer)

    # Save assistant message
    st.session_state.chat_history.append(("assistant", answer))
