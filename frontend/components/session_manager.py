"""
Session Management Component for Streamlit UI.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from backend.core.memory_manager import ResearchSession


def render_session_history(sessions: List[ResearchSession]):
    """
    Render session history in Streamlit.
    
    Args:
        sessions: List of research sessions
    """
    st.subheader("Research History")
    
    if not sessions:
        st.info("No previous research sessions.")
        return
    
    # Display recent sessions
    for session in sessions[:10]:  # Show last 10
        with st.expander(f"Query: {session.query[:50]}..."):
            st.write(f"**Session ID:** {session.id}")
            st.write(f"**Created:** {session.created_at}")
            st.write(f"**Updated:** {session.updated_at}")
            
            if session.report:
                st.write("**Report Preview:**")
                st.markdown(session.report[:500] + "..." if len(session.report) > 500 else session.report)
            
            if session.quality_scores:
                st.write("**Quality Scores:**")
                for metric, score in session.quality_scores.items():
                    st.write(f"- {metric}: {score:.1f}/10")


def render_session_selector(sessions: List[ResearchSession]) -> Optional[str]:
    """
    Render session selector dropdown.
    
    Args:
        sessions: List of research sessions
        
    Returns:
        Selected session ID or None
    """
    if not sessions:
        return None
    
    session_options = {
        f"{s.query[:50]}... ({s.created_at[:10]})": s.id
        for s in sessions
    }
    
    selected = st.selectbox(
        "Load Previous Session",
        options=["None"] + list(session_options.keys()),
        key="session_selector"
    )
    
    if selected and selected != "None":
        return session_options[selected]
    
    return None

