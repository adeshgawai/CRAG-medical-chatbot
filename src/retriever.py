from src.helper import embedding_creation
from langchain_community.vectorstores import FAISS
_retriever = None

def return_retriever():
    global _retriever

    if _retriever is not None:
        return _retriever
    
    VECTOR_STORE_PATH = "faiss_index"
    embeddings = embedding_creation()
    
    # Load the local FAISS index
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True 
    )
    
    _retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})
    return _retriever