from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
# Get the absolute path using os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "cover_letter.pdf")
persist_directory = os.path.join(current_dir, "..", "data", "db")

def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separator="\n")
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store():
    docs = load_pdf(file_path)
    vectordb = Chroma.from_documents(
        documents=docs, 
        embedding = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001'), 
        persist_directory=persist_directory
    )
    # vectordb.persist()
    return vectordb