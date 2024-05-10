from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings

class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name_or_path: str, tokenizer: AutoTokenizer, model_kwargs: dict = None):
        super().__init__(model_name_or_path, model_kwargs=model_kwargs)
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")["input_ids"]
