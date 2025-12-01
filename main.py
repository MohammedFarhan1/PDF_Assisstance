import os
import asyncio
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tempfile

def initialize_session():
    st.session_state.chat_history = []
    st.session_state.file_processed = False
    st.session_state.last_input = None
    
# Page Configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update the Custom CSS section with the following:
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #FF9900;
            color: white;
        }
        .stButton > button:hover {
            background-color: #FF8000;
            color: white;
        }
        .upload-text {
            text-align: center;
            padding: 2rem;
            border: 2px dashed #FF9900;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            background-color: #f0f2f6;
        }
        /* Fix for white text input */
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
            color: #000000 !important;
        }
        /* Additional styling for better input visibility */
        .stTextInput {
            color: #000000;
        }
        input {
            color: #000000 !important;
        }
        .st-bw {
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = None


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/chat.png", width=100)
    st.title("PDF Chat Assistant")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload your PDF document
    2. Wait for processing
    3. Ask questions about your document
    4. Get AI-powered responses
    """)
    st.markdown("---")
    if st.button("Clear Chat History"):
        initialize_session()

# Main content
load_dotenv()

# Ensure the Groq API key is set
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Groq API Key is missing! Set it in your .env file or environment variables.")
    st.stop()

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not st.session_state.file_processed:
        st.markdown("""
            <div class="upload-text">
                <h2>📄 Upload Your PDF Document</h2>
                <p>Upload a PDF file to start chatting about its contents</p>
            </div>
        """, unsafe_allow_html=True)
        
    uploaded_file = st.file_uploader("", type="pdf")

# Process the uploaded file
if uploaded_file is not None and not st.session_state.file_processed:
    with st.spinner("Processing your document... Please wait."):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Initialize embeddings and process document
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        loader = PyPDFLoader(pdf_path)
        st.session_state.docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Initialize LLM
        st.session_state.llm = ChatGroq(model_name="llama3-70b-8192", api_key=groq_api_key)
        
        # Create retrieval chain
        st.session_state.retriever = st.session_state.vector.as_retriever()
        
        prompt = ChatPromptTemplate.from_template("""
        Answer based on context: {context}
        Question: {question}
        """)
        
        st.session_state.retriever_chain = (
            {"context": st.session_state.retriever, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
        )
        
        st.session_state.file_processed = True
        os.unlink(pdf_path)
    
    st.success("✅ Document processed successfully!")

# Chat interface
if st.session_state.file_processed:
    st.markdown("### Chat with your PDF")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
            st.markdown("---")

    # Chat input
    user_prompt = st.text_input("Ask a question about your document:", key="user_input")
    
    if user_prompt and user_prompt != st.session_state.last_input:
        with st.spinner("Thinking..."):
            response = st.session_state.retriever_chain.invoke(user_prompt)
            
            # Add to chat history only if it's a new input
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Update last input
            st.session_state.last_input = user_prompt
            
            st.rerun()

# Clear chat history function
def initialize_session():
    st.session_state.chat_history = []
    st.session_state.file_processed = False
    st.session_state.last_input = None

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by Farhan</p>
    </div>
""", unsafe_allow_html=True)
