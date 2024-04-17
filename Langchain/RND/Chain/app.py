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
#print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

# ## Vector Embedding And Vector Store
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(documents,OllamaEmbeddings(model="llama2:13b"))

# query = "Why python is called an interpreted language?"
# retireved_results=db.similarity_search(query)
# print(retireved_results[0].page_content)

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

## Chain Introduction
## Create Stuff Docment Chain

from langchain_community.llms import Ollama
## Load Ollama LAMA2 LLM model
llm=Ollama(model="llama2:13b")

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt)

retriever=db.as_retriever()

from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)

response=retrieval_chain.invoke({"input":"Why python is called an interpreted language?"})

print(response['answer'])