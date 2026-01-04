from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain

import os
from typing import List, Tuple, Dict, Any
import re

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

class ReactDotChatbot:
    def __init__(
            self,
            db_path: str = './react_db',
            model_name: str = 'llama-3.1-8b-instant'      
    ):
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')


        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        self.llm = ChatGroq(model=model_name,temperature=0,max_tokens=500)
        prompt = ChatPromptTemplate.from_template(
            """You are a React expert assistant. Use the following pieces of retrieved context 
        from the official React documentation to answer the user's question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {input}
        
        Helpful Answer:"""
        )

        document_chain = create_stuff_documents_chain(self.llm,prompt)
        retriever = self.vector_db.as_retriever()

        self.retriever_chain = create_retrieval_chain(retriever,document_chain)

    def ask(self,query:str) -> Dict[str,Any]:
        result = self.retriever_chain.invoke({'input': query})

        answer: str = result["answer"]
        # Extract unique source file paths from metadata
        # sources: List[str] = list(set([
        #     doc.metadata.get("source", "Unknown") 
        #     for doc in result["source"]
        # ]))
        contexts: List[str] = []
        for context in result['context']:
            contexts.append(context.page_content)
        
        return {
            "answer": answer,
            "sources": contexts
        }
        # return {
        #     'answer': result['answer']
        # }

if __name__ == "__main__":
    bot = ReactDotChatbot()
    
    question = "can you help me with the code for useState hook?"
    response = bot.ask(question)
    
    print(f"Chatbot: {response['answer']}")
    print("\nSources used:")
    for src in response['sources']:
        print(f"- {src}")
    # print(response)