import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # To get embedding model.
from langchain.schema import Document # To store the text and metadata.
from langchain.text_splitter import CharacterTextSplitter # To split the raw text into chunks.
from langchain_community.vectorstores import FAISS # To store the embedded data for the similarity search.

key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)

gemini_model = genai.GenerativeModel('gemini-2.0-flash')


def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

with st.spinner('Loading Embedding Model...'):
    embedding_model = load_embedding()

st.header('RAG Assistant :blue[Using Embedding & Gemini LLM]')
st.subheader('Your intelligent Document Assistant!')


uploaded_file = st.file_uploader('Upload the document here in PDF format', type=['pdf'])

    
if uploaded_file:
    pdf = PdfReader(uploaded_file)
    raw_text = ''
    
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    
    if raw_text.strip():
        doc = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunk_text = splitter.split_documents([doc])
        
        text = [i.page_content for i in chunk_text]
        
        vector_db = FAISS.from_texts(text, embedding_model)
        retrive = vector_db.as_retriever()
        
        st.success('Document Processed Successfully... Ask your questions now')
        
        query = st.text_input('Enter your query here:')
        
        if query:
            with st.chat_message('human'):
                
                with st.spinner('Analyzing the Document'):
                    relevent_docs = retrive.get_relevant_documents(query)
                    
                    content = '\n\n'.join([i.page_content for i in relevent_docs])
                  
                    prompt = f''' 
                    You are an AI expert. Use the content given to answer the query asked by the user.
                    If you are unsure you should say 'I am unsure about the question asked'
                    
                    Content: {content}
                    
                    Query: {query}
                    
                    Result : '''
                    
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown('### :green[Result]')
                    st.write(response.text)
                    
    else:
        
        st.warning('Drop the file in proper PDF format')
                             
                             