#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
#from langchain_openai import OpenAIEmbeddings
#from chromadb.utils import embedding_functions
#from langchain_community.embeddings import SentenceTransformerEmbeddings

from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    #def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

def get_embedding_function():
    return LocalEmbeddings()
