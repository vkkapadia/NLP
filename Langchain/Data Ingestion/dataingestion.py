import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import shutil


def load_pdf(file_path):
    """Function to extract text from a PDF file using PyMuPDF."""
    # Open the PDF file
    loader=PyPDFLoader(file_path)
    pages =loader.load()
    text = []
    # Iterate through each page and extract text
    for doc in pages:
        text.append(doc.page_content)
    
    # Return the extracted text as a single string
    return "\n".join(text)

def extract_pdfs_from_folder(folder_path,documents):
    # Declare documents as global variable
    """Function to extract text from all PDFs in a specified folder."""
    pdf_files = []
    print('started pdfs')
    print('Folder Path ' + folder_path )
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            # Construct the full file path
            print('----------- Filename ' + filename)
            file_path = os.path.join(folder_path, filename)
            # Load the PDF and extract text
            pdf_text = load_pdf(file_path)
            pdf_files.append(file_path)
            # Append the extracted text to the list
            documents.append(pdf_text)
    return pdf_files, documents

def generate_chunks(documents):
    
    docs = []  # Declare documents as global variable

    # Define the text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Convert text to documents
    
    
    # for doc_text in documents:
    #     print('---' + doc_text)
    #     chunks = text_splitter.split_text(doc_text)
    #     docs += [Document(page_content=chunk) for chunk in chunks]

    chunks = text_splitter.split_text('My name is india')
    docs += [Document(page_content=chunk) for chunk in chunks]

    return docs

def update_vectorstore(docs):

    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL"))


    # Create or load your existing vector store (you may want to save and load it)
    # Load existing vector store (if you already have one)
    try:
        vector_store = FAISS.load_local(os.getenv("VECTOR_STORE_PATH"), embeddings)
         # Update the vector store with the new documents and their embeddings
        vector_store.add_documents(docs, embeddings)
    except:
        print('------------part 2')
        # If not found, create a new vector store
        vector_store = FAISS.from_documents(docs, embeddings)


   

    # Save the updated vector store for future use
    vector_store.save_local(os.getenv("VECTOR_STORE_PATH"))

def run_etl():
    # Enable dangerous deserialization
    print(1)
    load_dotenv()
    documents=[]
    print('1')
    pdf_files, documents = extract_pdfs_from_folder(os.getenv("DOCS_PATH"),documents)
    docs = generate_chunks(documents)
    print('print docs----------------')
    print(docs)
    update_vectorstore(docs)
    
    # # Move all PDF files from the source folder to the backup folder
    for pdf_file in pdf_files:
    #     # Move the PDF file to the backup path
        shutil.move(src=pdf_file, dst=os.path.join(os.getenv("DOCS_COMPLETED_PATH"), os.path.basename(pdf_file)))
