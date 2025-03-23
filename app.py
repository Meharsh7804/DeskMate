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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Update model name to use gemini-1.5-pro or gemini-1.0-pro instead of gemini-pro
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",  # Try this first
                                  temperature=0.3)
    
    # If gemini-1.5-pro doesn't work, use the fallback in the error handling section below

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Add allow_dangerous_deserialization=True to fix the previous error
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        try:
            chain = get_conversational_chain()
            
            response = chain(
                {"input_documents":docs, "question": user_question}
                , return_only_outputs=True)

            print(response)
            st.write("Reply: ", response["output_text"])
            
        except Exception as model_error:
            st.error(f"Error with model: {str(model_error)}")
            # Fallback to alternative model if first one fails
            st.warning("Trying with alternative model...")
            
            fallback_model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",  # Fallback model
                                         temperature=0.3)
            
            prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            fallback_chain = load_qa_chain(fallback_model, chain_type="stuff", prompt=prompt)
            
            response = fallback_chain(
                {"input_documents":docs, "question": user_question}
                , return_only_outputs=True)
                
            print(response)
            st.write("Reply: ", response["output_text"])
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please try uploading your PDF files first using the sidebar menu.")

def main():
    st.set_page_config("Chat PDF")
    st.header("DeskMate")

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()