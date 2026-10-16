# =============================================================================
# TODO(v2): RETIRED IN v2. Ingestion is split into small, testable pieces:
#   codesearch/ingest/fetch.py      react.dev at a pinned commit
#   codesearch/ingest/mdx_clean.py  clean one page (pure function)
#   codesearch/ingest/urls.py       file path -> react.dev URL + anchor
#   codesearch/ingest/chunker.py    heading-aware chunks with stable IDs
#   codesearch/ingest/manifest.py   what the index was built from
#   codesearch/ingest/build.py      end-to-end, incremental
#   codesearch/index/embedder.py, vector_store.py, bm25.py
# =============================================================================

# Description: Importing all necessary documents
# TODO(v2): unused import.
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# TODO(v2): tqdm is not in requirements.txt.
from tqdm import tqdm  # <--- Added for progress bar
from dotenv import load_dotenv
import os
from typing import List, Tuple
import re

# Description: Setting up the environment variables
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# TODO(v2): BUG: os.environ[...] = None raises TypeError when the variable is unset.
#     Optional LangSmith tracing becomes a setting in config.py.
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
    # TODO(v2): depends on the consolidated_react_docs.md dump. v2 reads pages directly from
    #     a react.dev checkout at a pinned commit (ingest/fetch.py); no dump file.
    file_sections: List[str] = re.split(r'--- SOURCE: (.*?) ---', content)
    raw_documents: List[Document] = []
    
    for i in range(1, len(file_sections), 2):
        source_name: str = file_sections[i].strip()
        text_content: str = file_sections[i+1].strip()
        # TODO(v2): this is the only MDX cleanup. ingest/mdx_clean.py must also unwrap <Sandpack>
        #     into code fences, label <Pitfall>/<Note>, and parse the frontmatter title.
        text_content = re.sub(r'\{/\*.*?\*/\}', '', text_content)
        
        raw_documents.append(Document(
            page_content=text_content,
            # TODO(v2): stores a local path. v2 Chunk (schemas.py) stores url, anchor, header_path.
            metadata={"source": source_name}
        ))

    # 3. STRUCTURAL SPLITTING
    headers_to_split_on: List[Tuple[str, str]] = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # TODO(v2): the header path only lands in metadata, so the embedder never sees it.
    #     v2 chunker prepends it to the chunk text and never splits inside a code fence.
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
        # TODO(v2): 1000 characters, not tokens (the README says tokens). v2: IndexSettings.max_chars.
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
    # TODO(v2): no stable IDs, so re-running adds duplicate chunks. v2: chunk_id =
    #     sha256(url + header_path + text); diff against the live index, embed only new
    #     chunks, upsert/delete, and write index/manifest.json.
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
        # TODO(v2): manual deletion goes away; ingest/build.py handles changes by diffing IDs.
        # Note: If you want to start fresh, delete the ./react_db folder first
        db = create_react_vector_store(INPUT_FILE, DB_DIRECTORY)
    except Exception as err:
        print(f"An error occurred during vector store creation: {err}")