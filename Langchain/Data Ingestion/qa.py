from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import os
import asyncio
from dotenv import load_dotenv
import faiss
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
     llm = Ollama(model=os.getenv("OLLAMA_MODEL"))
     # Load the existing vector store
     embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL"))
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
         "What is your name?"
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
        # Create a standard L2 index on the CPU
    index_cpu = faiss.IndexFlatL2(128)

    # Create GPU resources
    gpu_res = faiss.StandardGpuResources()

    # Move the index to GPU, specifying the GPU device ID (0 for first GPU)
    index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
    asyncio.run(main=main())