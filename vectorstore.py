# Description: Importing all necessary documents
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm  # <--- Added for progress bar
from dotenv import load_dotenv
import os
from typing import List, Tuple
import re

# Description: Setting up the environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Descsription: Loading the data and creating the vector store
def create_react_vector_store(
    file_path: str, 
    db_path: str = "./react_db",
    batch_size: int = 50  # Added batch_size parameter
) -> Chroma:
    """
    Parses a consolidated markdown file, splits it into semantic chunks 
    preserving metadata, and persists it into a Chroma vector database.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # 1. READ CONTENT
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content: str = f.read()
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}")

    # 2. SEMANTIC FILING
    file_sections: List[str] = re.split(r'--- SOURCE: (.*?) ---', content)
    raw_documents: List[Document] = []
    
    for i in range(1, len(file_sections), 2):
        source_name: str = file_sections[i].strip()
        text_content: str = file_sections[i+1].strip()
        text_content = re.sub(r'\{/\*.*?\*/\}', '', text_content)
        
        raw_documents.append(Document(
            page_content=text_content,
            metadata={"source": source_name}
        ))

    # 3. STRUCTURAL SPLITTING
    headers_to_split_on: List[Tuple[str, str]] = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )

    all_header_splits: List[Document] = []
    for doc in raw_documents:
        header_splits: List[Document] = header_splitter.split_text(doc.page_content)
        for split in header_splits:
            split.metadata.update(doc.metadata)
            all_header_splits.append(split)

    # 4. GRANULAR SPLITTING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True 
    )
    
    final_docs: List[Document] = text_splitter.split_documents(all_header_splits)

    # 5. EMBEDDING & PERSISTENCE (With tqdm progress bar)
    print(f"Total chunks to process: {len(final_docs)}")
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    
    # Initialize Chroma with the first batch
    first_batch = final_docs[:batch_size]
    vector_db: Chroma = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # Process the remaining batches with a progress bar
    for i in tqdm(range(batch_size, len(final_docs), batch_size), desc="Embedding React Docs"):
        batch = final_docs[i : i + batch_size]
        vector_db.add_documents(batch)
    
    print(f"\n✅ Successfully processed {len(final_docs)} chunks into {db_path}")
    return vector_db

if __name__ == "__main__":
    INPUT_FILE: str = "consolidated_react_docs.md"
    DB_DIRECTORY: str = "./react_db"
    
    try:
        # Note: If you want to start fresh, delete the ./react_db folder first
        db = create_react_vector_store(INPUT_FILE, DB_DIRECTORY)
    except Exception as err:
        print(f"An error occurred during vector store creation: {err}")