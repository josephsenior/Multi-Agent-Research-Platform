"""
Enhanced RAG System for the Multi-Agent Research Platform.

This extends the basic RAG pattern with:
- Document metadata tracking
- Persistent vector store
- Better chunking strategies
- Metadata filtering
- Integration with document management
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.schema.runnable import RunnablePassthrough

load_dotenv()


class RAGSystem:
    """
    Enhanced RAG system for document retrieval and question answering.
    
    Features:
    - Persistent vector store
    - Metadata tracking (source, page, document_id)
    - Better chunking with RecursiveCharacterTextSplitter
    - Metadata filtering
    - Batch document loading
    """
    
    def __init__(
        self,
        vectorstore_path: Optional[str] = None,
        model_name: str = "gpt-4",
        temperature: float = 0,
        api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG System.
        
        Args:
            vectorstore_path: Path to save/load vector store
            model_name: OpenAI model name
            temperature: LLM temperature
            api_key: OpenAI API key
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vectorstore = None
        self.retriever = None
        self.vectorstore_path = vectorstore_path or "data/vectorstore"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Ensure vectorstore directory exists
        Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
        
        # Document metadata tracking
        self.document_metadata = {}
        
        self._build_rag_chain()
    
    def _build_rag_chain(self):
        """Build the RAG chain."""
        # Enhanced RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant that answers questions based on 
the provided context from documents. Use only the information from the context to answer questions. 
If the context doesn't contain enough information to answer the question, say so.
Always cite your sources when providing information.

Context:
{context}

Answer the question based on the context above. Include relevant citations."""),
            ("user", "{question}")
        ])
        
        # RAG chain will be built after documents are loaded
        self.rag_chain = None
    
    def load_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Load and process documents for retrieval.
        
        Args:
            documents: List of document texts
            metadata_list: Optional list of metadata dicts for each document
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        # Convert to Document objects with metadata
        doc_objects = []
        for i, (doc_text, metadata) in enumerate(zip(documents, metadata_list)):
            doc_metadata = {
                "document_id": metadata.get("document_id", f"doc_{i}"),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", None),
                "title": metadata.get("title", f"Document {i+1}"),
                **metadata
            }
            doc_objects.append(Document(page_content=doc_text, metadata=doc_metadata))
        
        # Split documents into chunks with better strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(doc_objects)
        
        # Store metadata
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id not in self.document_metadata:
                self.document_metadata[doc_id] = chunk.metadata
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        
        # Create retriever with metadata filtering capability
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Build RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"Loaded {len(chunks)} document chunks into vector store")
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for the prompt with metadata."""
        formatted = []
        for doc in docs:
            source_info = f"[Source: {doc.metadata.get('source', 'unknown')}"
            if doc.metadata.get('page'):
                source_info += f", Page {doc.metadata['page']}"
            source_info += "]"
            formatted.append(f"{source_info}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    def query(
        self,
        question: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            Dictionary with answer, retrieved context, and sources
        """
        if not self.rag_chain:
            raise ValueError("Documents must be loaded first. Call load_documents()")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Update retriever k if needed
        if k != 5:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            self.rag_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)
        
        # Generate answer
        answer = self.rag_chain.invoke(question)
        
        # Extract sources
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "document_id": doc.metadata.get("document_id")
            })
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": [doc.page_content for doc in retrieved_docs],
            "sources": sources,
            "num_retrieved": len(retrieved_docs)
        }
    
    def add_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add more documents to the existing knowledge base.
        
        Args:
            documents: List of new document texts
            metadata_list: Optional list of metadata dicts
        """
        if not self.vectorstore:
            self.load_documents(documents, metadata_list)
            return
        
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        # Convert to Document objects
        doc_objects = []
        for i, (doc_text, metadata) in enumerate(zip(documents, metadata_list)):
            doc_metadata = {
                "document_id": metadata.get("document_id", f"doc_{len(self.document_metadata) + i}"),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", None),
                "title": metadata.get("title", f"Document {len(self.document_metadata) + i + 1}"),
                **metadata
            }
            doc_objects.append(Document(page_content=doc_text, metadata=doc_metadata))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(doc_objects)
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        
        # Update metadata
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id not in self.document_metadata:
                self.document_metadata[doc_id] = chunk.metadata
        
        print(f"Added {len(chunks)} new document chunks")
    
    def save_vectorstore(self):
        """Save the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(self.vectorstore_path)
            # Save metadata
            metadata_path = Path(self.vectorstore_path) / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.document_metadata, f, indent=2)
            print(f"Vector store saved to {self.vectorstore_path}")
    
    def load_vectorstore(self):
        """Load the vector store from disk."""
        vectorstore_file = Path(self.vectorstore_path) / "index.faiss"
        if vectorstore_file.exists():
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Load metadata
            metadata_path = Path(self.vectorstore_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.document_metadata = json.load(f)
            
            # Rebuild RAG chain
            self.rag_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            print(f"Vector store loaded from {self.vectorstore_path}")
        else:
            print("No existing vector store found. Create one by loading documents.")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        return len(self.document_metadata) if self.document_metadata else 0

