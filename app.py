import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure page
st.set_page_config(
    page_title="DeskMate AI | PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    .message-content {
        margin-left: 1rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .user-avatar {
        background-color: #1E88E5;
        color: white;
    }
    .bot-avatar {
        background-color: #4CAF50;
        color: white;
    }
    .file-list {
        background-color: #ECEFF1;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .file-item {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        border-bottom: 1px solid #CFD8DC;
    }
    .file-icon {
        margin-right: 0.5rem;
        color: #1E88E5;
    }
    .stTextInput>div>div>input {
        background-color: #F5F5F5;
        border-radius: 25px;
        padding: 1rem;
        border: none;
        box-shadow: none;
    }
    .help-text {
        font-size: 0.8rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'ready_to_chat' not in st.session_state:
    st.session_state.ready_to_chat = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are DeskMate, a helpful and knowledgeable AI assistant specialized in retrieving information from documents.
    
    Answer the question based on the provided context. Be detailed, accurate, and helpful.
    If the answer is not found in the context, clearly state "I don't have enough information in the documents to answer this question" and suggest what additional information might help.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """

    # Try to use gemini-1.5-pro first
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    except:
        # Fall back to gemini-1.0-pro if needed
        model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def process_query(user_question):
    if not st.session_state.ready_to_chat:
        st.session_state.chat_history.append({"role": "assistant", "content": "Please upload and process some PDF documents first."})
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Create a placeholder for the assistant's message with loading animation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("🤔 Thinking...")
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)
            
            try:
                chain = get_conversational_chain()
                response = chain(
                    {"input_documents": docs, "question": user_question},
                    return_only_outputs=True
                )
                answer = response["output_text"]
            except Exception as model_error:
                st.error(f"Error with model: {str(model_error)}")
                answer = "I encountered an issue while processing your question. Let me try an alternative approach."
                
                # Fallback to alternative model
                fallback_model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
                prompt_template = """
                Answer the question based on the provided context. Be detailed, accurate, and helpful.
                If the answer is not in the context, clearly state "I don't have enough information in the documents to answer this question."
                
                Context:
                {context}
                
                Question: 
                {question}
                
                Answer:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                fallback_chain = load_qa_chain(fallback_model, chain_type="stuff", prompt=prompt)
                
                response = fallback_chain(
                    {"input_documents": docs, "question": user_question},
                    return_only_outputs=True
                )
                answer = response["output_text"]
            
            # Update the placeholder with the final answer
            message_placeholder.empty()
            message_placeholder.write(answer)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.empty()
            message_placeholder.write(f"❌ {error_message}")
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

def process_documents(pdf_docs):
    if not pdf_docs:
        st.warning("Please upload at least one PDF document.")
        return
    
    with st.spinner("🔎 Reading documents..."):
        raw_text = get_pdf_text(pdf_docs)
        st.session_state.processed_files = [doc.name for doc in pdf_docs]
    
    with st.spinner("✂️ Splitting into chunks..."):
        text_chunks = get_text_chunks(raw_text)
    
    with st.spinner("🧠 Creating knowledge base..."):
        get_vector_store(text_chunks)
    
    st.session_state.ready_to_chat = True
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": f"✅ I've processed {len(pdf_docs)} document(s). You can now ask me questions about them!"
    })

def main():
    # Two-column layout: Sidebar and Main content
    with st.sidebar:
        st.markdown('<div class="main-header">DeskMate AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Your PDF Knowledge Assistant</div>', unsafe_allow_html=True)
        
        st.markdown("### 📚 Document Library")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload your PDF files", 
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files at once"
        )
        
        process_btn = st.button("Process Documents")
        if process_btn and pdf_docs:
            process_documents(pdf_docs)
        
        if st.session_state.processed_files:
            st.markdown("### 📋 Processed Files")
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            for file in st.session_state.processed_files:
                st.markdown(f'<div class="file-item"><span class="file-icon">📄</span> {file}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="help-text">DeskMate AI helps you extract insights from your PDF documents using advanced AI technology.</p>', unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="main-header">Ask DeskMate</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    if not st.session_state.ready_to_chat:
        st.info("👋 Hello! Please upload and process some PDF documents to get started.")
    
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        process_query(user_question)

if __name__ == "__main__":
    main()