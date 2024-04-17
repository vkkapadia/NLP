## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

loader=PyPDFLoader('1.pdf')
docs=loader.load()
print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

# ## Vector Embedding And Vector Store
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(documents,OllamaEmbeddings(model="llama2:13b"))

query = "Why python is called an interpreted language?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)