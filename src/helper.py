from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# import txtai
# from langchain_community.embeddings import FastEmbedEmbeddings

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_spltter(extracted_docs):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_split.split_documents(extracted_docs)
    return text_chunks


# def embedding_creation():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     return embeddings

def embedding_creation():
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    return embeddings