import os

import asyncio
import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Load environment variables from .env file
load_dotenv()

prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
Example of your response should be as follows:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def initialize_qa_chain():
     # Initialize language model
    llm = HuggingFacePipeline.from_model_id(
         model_id="sentence-transformers/all-MiniLM-L6-v2",
         task="text-generation",
         pipeline_kwargs={"temperature": 0.7, "do_sample": True, "max_length":500},
     )
    
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    device="cpu"
     # Load the existing vector store
    embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
     )

    vector_store = FAISS.load_local(os.getenv("VECTOR_STORE_PATH"), embeddings,  allow_dangerous_deserialization=True)
     # Create a RetrievalQA chain using the vector store and language model
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
     )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        )
    return qa_chain
async def ask_questions(qa_chain):
     
     # Ask questions and get answers
     questions = [
         "What speacker want to tell you?"
     ]
     for question in questions:
         try:
             print("question asked " +  question)
             # Get the answer from the QA chain
             answer =await qa_chain.ainvoke({'query':question})
             print(f"Question: {question}")
             print(f"Answer: {answer}")
         except Exception as e:
             print(f"Error processing question '{question}': {e}")
async def main():
     # Initialize QA chain
     qa_chain = initialize_qa_chain()
     # Ask questions and get answers
     await ask_questions(qa_chain)
if __name__ == "__main__":
    asyncio.run(main=main())