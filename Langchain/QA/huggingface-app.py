import os
import torch

import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
#from transformers import AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables from .env file
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(HUGGINGFACEHUB_API_TOKEN)
DB_FAISS_PATH = '/content/drive/MyDrive/VectorStore/'
print(DB_FAISS_PATH)
MODEL_PATH = "C:\\Users\\vriha\\.cache\\huggingface\\hub\\models--meta-llama--Meta-Llama-3-8B\\"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt_template = prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def create_retrieval_qa_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,

        chain_type="stuff",
        retriever=db.as_retriever(  search_kwargs={"k":1}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def load_model(
    model_path=MODEL_PATH,
    model_type="llama",
    max_new_tokens=512,
    temperature=0.7,
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    #if not os.path.exists(model_path):
    #   raise FileNotFoundError(f"No model file found at {model_path}")

    # Additional error handling could be added here for corrupt files, etc.

    #llm = CTransformers.AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", hf=True)

    # llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    #     #Hugging Face models can be run locally through the HuggingFacePipeline class.
    

    print('---before downloading llam3--')
    
    # llm = HuggingFacePipeline.from_model_id(
    #      model_id="sentence-transformers/all-MiniLM-L6-v2",
    #      task="text-generation",
    #      pipeline_kwargs={"temperature": temperature, "do_sample": True, "max_length": max_new_tokens},
    #  )
    # llm  = HuggingFacePipeline.from_model_id(
    # model_id="meta-llama/Meta-Llama-3-8B",
    # task="text-generation",
    # device=0,
    # pipeline_kwargs={"temperature": temperature, "do_sample": True, "max_length": max_new_tokens},
    # )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_4bit = AutoModelForCausalLM.from_pretrained( "mistralai/Mistral-7B-Instruct-v0.1", device_map="auto",quantization_config=quantization_config, )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    pipeline_inst = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipeline_inst)
    #llm_model = AutoModelForSeq2SeqLM.from_pretrained("mistralai/Mistral-7B-v0.1")


    #llm = hf 

    return llm


def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    persist_dir=DB_FAISS_PATH,
    device="cuda",
):
    """
    This function creates a retrieval-based question-answering bot.

    Parameters:
        model_name (str): The name of the model to be used for embeddings.
        persist_dir (str): The directory to persist the database.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        SomeOtherException: If there is an issue with loading the embeddings or the model.
    """

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        print('-----------step-1---------------')
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )

        print('-----------step-2---------------')
        #embeddings.client.tokenizer.pad_token =  embeddings.client.tokenizer.eos_token

    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    print('-----------step-3---------------')
    db = FAISS.load_local(folder_path=DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

    try:
        print('-----------step-4---------------')
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:

        qa = create_retrieval_qa_chain(
                 llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


def retrieve_bot_answer(query):
    """
    Retrieves the answer to a given query using a QA bot.

    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.

    Args:
        query (str): The question to be answered by the QA bot.

    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance({"query": query})
    return bot_response


@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    print('new started 1')
    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llama2 and LangChain."
    )
    await welcome_message.update()

    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def process_chat_message(message: cl.Message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    print(f"res : {message.content}")
    response = await qa_chain.ainvoke(message.content, callbacks=[callback_handler])
    print(f"--------------------------response : {response}")
        # Extract bot's answer from the response
    bot_answer = response["result"]
    source_documents = response["source_documents"]

    if source_documents:
        bot_answer += f"\nSources:" + str(source_documents)
    else:
        bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()