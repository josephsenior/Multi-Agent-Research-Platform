"""
Document Upload Component for Streamlit UI.
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
from backend.tools.pdf_parser import PDFParser


def render_document_upload() -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Render document upload component.
    
    Returns:
        Tuple of (document texts, metadata list)
    """
    st.subheader("Upload Documents")
    st.write("Upload PDF documents to include in your research.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    documents = []
    metadata_list = []
    pdf_parser = PDFParser()
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                upload_dir = Path("data/documents")
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Parse PDF
                parsed = pdf_parser.parse_file(str(file_path), extract_metadata=True)
                
                documents.append(parsed["text"])
                metadata_list.append({
                    "document_id": uploaded_file.name,
                    "source": uploaded_file.name,
                    "title": parsed["metadata"].get("title") or uploaded_file.name,
                    "author": parsed["metadata"].get("author"),
                    "num_pages": parsed["num_pages"]
                })
                
                st.success(f"Loaded: {uploaded_file.name} ({parsed['num_pages']} pages)")
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return documents, metadata_list

