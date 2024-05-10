import os
from dotenv import load_dotenv
import torch
#import textwrap

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from transformers import AutoTokenizer
from CustomEmbeddings import CustomHuggingFaceEmbeddings
# Load environment variables from .env file
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(HUGGINGFACEHUB_API_TOKEN)
DB_FAISS_PATH = os.getenv("VECTOR_STORE_PATH")
DATA_DIR = os.getenv("DOCS_PATH")



# def wrap_text_preserve_newlines(text, width=110):
#     # Split the input text into lines based on newline characters
#     lines = text.split('\n')
#     # Wrap each line individually
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
#     # Join the wrapped lines back together using newline characters
#     wrapped_text = '\n'.join(wrapped_lines)
#     return wrapped_text

# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Initialize loaders for different file types
    """ pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    markdown_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)

    all_loaders = [pdf_loader, markdown_loader, text_loader]

    # Load documents from all loaders
    loaded_documents = []
    for loader in all_loaders:
        loaded_documents.extend(loader.load()) """
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    #text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    loaded_documents =pdf_loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents(loaded_documents)


#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B")

#     # Set padding token
#     tokenizer.pad_token = tokenizer.eos_token  # For example, using end-of-sequence token as padding


#   # Initialize HuggingFace embeddings
#     huggingface_embeddings = CustomHuggingFaceEmbeddings(
#         model_name_or_path="meta-llama/Meta-Llama-3-8B",
#         tokenizer=tokenizer,
#         model_kwargs={"device": "cpu"},
#     )
    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    
    #huggingface_embeddings.client.tokenizer.pad_token =  huggingface_embeddings.client.tokenizer.eos_token


    # Create and persist a Chroma vector database from the chunked documents
    vector_database = FAISS.from_documents(
        documents=chunked_documents,
        embedding=huggingface_embeddings,

    )

    vector_database.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    #print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    create_vector_database()