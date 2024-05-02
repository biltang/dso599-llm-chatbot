import os 
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv('../.env')


def get_embedding_function(chosen_embedding: str="openai"):
    """Given a chosen embedding, return the corresponding embedding function 
    from the embedding_functions dictionary.

    Args:
        chosen_embedding (str, optional): which embedding function to choose from embedding_functions dictionary. Defaults to "openai".

    Returns:
        an embedding function be used for constructing a vector store.
    """
    embedding_functions = {
        "openai": OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    }
    return embedding_functions['openai']


def load_documents(data_path: str) -> list[Document]:
    """ Load documents from a directory of PDFs.

    Args:
        data_path (str): directory path containing PDFs.

    Returns:
        list[Document]: list of documents loaded from the directory.
    """
    loader = PyPDFDirectoryLoader(data_path, recursive=True)
    return loader.load()


def split_documents(documents: list[Document], chunk_size: int=1000, chunk_overlap: int=200) -> list:
    """ Split documents into chunks of text.

    Args:
        documents (list[Document]): list of documents to split.
        chunk_size (int, optional): size of each chunk to split documents into. Defaults to 1000.
        chunk_overlap (int, optional): how much overlap the chunks have. Defaults to 200.

    Returns:
        list: list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_brand_new_chroma_db(chunks: list, db_path: str) -> None:
    """ Create a new Chroma DB from loaded chunks.

    Args:
        chunks (list): list of text chunks.
        db_path (str): path to save the Chroma DB.
    """
    # remove existing Chroma DB if it exists if we are creating a brand new one
    if os.path.exists(db_path):
        print('removing existing Chroma DB...')
        shutil.rmtree(db_path)
    
    # create new Chroma DB from loaded chunks
    vector_store = Chroma.from_documents(chunks, get_embedding_function(), persist_directory=db_path)
    vector_store.persist()
    print(f"Chroma DB created at {db_path} with {len(chunks)} chunks.")
    

def create_retriver_tool_chroma_db(db_path: str) -> None:
    """ Create a retriever tool from a Chroma DB.

    Args:
        db_path (str): path to the Chroma DB.
    """
    vector_store = Chroma(persist_directory=db_path, embedding_function=get_embedding_function())
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(retriever,
                                        name="dinosaur_temperature_transport_info",
                                        description="""Based on dinosaur name,
                                        searches and returns information about safe temperatures 
                                        and conditions for transporting dinosaurs and the 
                                        responsibilities of DINO transport team members. Also contains
                                        actions to take to keep Dino safe if the temperature during
                                        transportation is outside the safe range.""")
    
    return retriever_tool


def create_chroma_db_vector_store(doc_dir: str, db_path: str):
    """Wrapper function to create a chromadb vector store from doc_dir. Allows us
    to create a chromadb vector store from scratch from a central application.

    Args:
        doc_dir (str): document path to load documents from
        db_path (str): path to store chromadb vector store
    """
    documents = load_documents(doc_dir) # load documents from directory
    chunks = split_documents(documents) # split documents into chunks
    create_brand_new_chroma_db(chunks, db_path=db_path) # create a new Chroma DB
    
    
def main():
    
    create_chroma_db_vector_store(doc_dir='../../pdf_data/', db_path='../../chroma_vector_store/')
    
    
if __name__ == '__main__':
   main()