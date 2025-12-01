"""
Main Streamlit Application for the Multi-Agent Research Platform.
"""

import streamlit as st
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.orchestrator import ResearchOrchestrator
from backend.core.rag_system import RAGSystem
from backend.tools.web_search import WebSearchTool
from backend.tools.pdf_parser import PDFParser
from frontend.components.document_upload import render_document_upload
from frontend.components.research_display import render_research_results
from frontend.components.session_manager import render_session_history, render_session_selector

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "orchestrator" not in st.session_state:
    try:
        st.session_state.orchestrator = ResearchOrchestrator()
        st.session_state.documents_loaded = False
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {str(e)}")
        st.stop()

if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None


def main():
    """Main application function."""
    st.title("Multi-Agent Research Platform")
    st.markdown("Conduct comprehensive research using multiple specialized AI agents.")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Research", "Documents", "History"],
            key="page_selector"
        )
        
        st.divider()
        
        # Display recent sessions
        if st.session_state.orchestrator:
            recent_sessions = st.session_state.orchestrator.get_recent_sessions(5)
            if recent_sessions:
                st.subheader("Recent Sessions")
                for session in recent_sessions:
                    st.write(f"- {session.query[:40]}...")
    
    # Main content area
    if page == "Research":
        render_research_page()
    elif page == "Documents":
        render_documents_page()
    elif page == "History":
        render_history_page()


def render_research_page():
    """Render the main research page."""
    st.header("Conduct Research")
    
    # Load previous session if selected
    if st.session_state.orchestrator:
        recent_sessions = st.session_state.orchestrator.get_recent_sessions(10)
        selected_session_id = render_session_selector(recent_sessions)
        
        if selected_session_id and selected_session_id != st.session_state.current_session_id:
            session = st.session_state.orchestrator.get_session(selected_session_id)
            if session:
                st.session_state.research_result = {
                    "query": session.query,
                    "report": session.report,
                    "quality_scores": session.quality_scores,
                    "citations": session.citations
                }
                st.session_state.current_session_id = selected_session_id
    
    # Research query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Research Query",
            value=st.session_state.research_result.get("query", "") if st.session_state.research_result else "",
            placeholder="Enter your research question...",
            key="research_query"
        )
    
    with col2:
        use_web_search = st.checkbox("Web Search", value=True, key="use_web_search")
        use_rag = st.checkbox("Document Search", value=True, key="use_rag")
    
    # Research button
    if st.button("Start Research", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a research query.")
            return
        
        with st.spinner("Conducting research... This may take a few minutes."):
            try:
                result = st.session_state.orchestrator.research(
                    query=query,
                    use_web_search=use_web_search,
                    use_rag=use_rag,
                    save_session=True
                )
                
                st.session_state.research_result = result
                st.session_state.current_session_id = result.get("session_id")
                st.success("Research complete!")
            
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
                st.exception(e)
                
                # Provide helpful error message
                error_msg = str(e).lower()
                if "api key" in error_msg:
                    st.info("Tip: Make sure your OpenAI API key is set in the .env file.")
                elif "timeout" in error_msg:
                    st.info("Tip: The request timed out. Try again or simplify your query.")
                elif "rate limit" in error_msg:
                    st.info("Tip: API rate limit exceeded. Please wait a moment and try again.")
    
    # Display results
    if st.session_state.research_result:
        st.divider()
        render_research_results(st.session_state.research_result)
        
        # Export options
        st.divider()
        st.subheader("Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as Markdown", use_container_width=True):
                markdown_content = f"# Research Report\n\n{st.session_state.research_result.get('report', '')}"
                st.download_button(
                    label="Download Markdown",
                    data=markdown_content,
                    file_name="research_report.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("Export Citations", use_container_width=True):
                from backend.core.citation_manager import CitationManager
                citation_manager = CitationManager()
                citation_manager.citations = st.session_state.research_result.get("citations", [])
                citations_text = citation_manager.generate_reference_list("apa")
                st.download_button(
                    label="Download Citations",
                    data=citations_text,
                    file_name="citations.txt",
                    mime="text/plain"
                )


def render_documents_page():
    """Render the documents management page."""
    st.header("Document Management")
    
    # Upload documents
    documents, metadata_list = render_document_upload()
    
    if documents:
        if st.button("Load Documents into RAG System", type="primary"):
            with st.spinner("Loading documents..."):
                try:
                    st.session_state.orchestrator.load_documents(documents, metadata_list)
                    st.session_state.documents_loaded = True
                    st.success(f"Loaded {len(documents)} documents into the RAG system!")
                except Exception as e:
                    st.error(f"Error loading documents: {str(e)}")
                    error_msg = str(e).lower()
                    if "pdf" in error_msg or "parse" in error_msg:
                        st.info("Tip: Make sure the PDF file is not corrupted or password-protected.")
                    elif "vector" in error_msg or "embedding" in error_msg:
                        st.info("Tip: Check your OpenAI API key is set correctly.")
        
        # Display loaded documents info
        if st.session_state.documents_loaded:
            st.info(f"{len(documents)} documents are available for research.")
    
    # Document statistics
    if st.session_state.orchestrator and st.session_state.orchestrator.rag_system:
        doc_count = st.session_state.orchestrator.rag_system.get_document_count()
        if doc_count > 0:
            st.metric("Documents in RAG System", doc_count)


def render_history_page():
    """Render the history page."""
    st.header("Research History")
    
    if st.session_state.orchestrator:
        all_sessions = st.session_state.orchestrator.get_all_sessions()
        render_session_history(all_sessions)
        
        # Session actions
        if all_sessions:
            st.divider()
            st.subheader("Session Actions")
            
            session_ids = {f"{s.query[:50]}...": s.id for s in all_sessions}
            selected_session = st.selectbox(
                "Select Session",
                options=["None"] + list(session_ids.keys()),
                key="history_session_selector"
            )
            
            if selected_session and selected_session != "None":
                session_id = session_ids[selected_session]
                session = st.session_state.orchestrator.get_session(session_id)
                
                if session:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Load Session", use_container_width=True):
                            st.session_state.research_result = {
                                "query": session.query,
                                "report": session.report,
                                "quality_scores": session.quality_scores,
                                "citations": session.citations
                            }
                            st.session_state.current_session_id = session.id
                            st.success("Session loaded! Switch to Research page to view.")
                    
                    with col2:
                        if st.button("Delete Session", use_container_width=True):
                            st.session_state.orchestrator.memory_manager.delete_session(session_id)
                            st.success("Session deleted!")
                            st.rerun()


if __name__ == "__main__":
    main()

