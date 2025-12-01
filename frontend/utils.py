"""
Utility functions for the Streamlit UI.
"""

import streamlit as st
from typing import Dict, Any, List


def format_citation(citation: Dict[str, Any], format_type: str = "apa") -> str:
    """
    Format a citation for display.
    
    Args:
        citation: Citation dictionary
        format_type: Citation format (apa, mla, chicago)
        
    Returns:
        Formatted citation string
    """
    if citation.get("type") == "web":
        url = citation.get("url", "")
        title = citation.get("title", "Untitled")
        return f"[{title}]({url})"
    elif citation.get("type") == "document":
        title = citation.get("title", "Document")
        page = citation.get("page")
        if page:
            return f"{title} (p. {page})"
        return title
    return str(citation)


def display_quality_scores(scores: Dict[str, float]):
    """
    Display quality scores in Streamlit.
    
    Args:
        scores: Dictionary of quality scores
    """
    if not scores:
        st.info("No quality scores available.")
        return
    
    cols = st.columns(3)
    metric_names = list(scores.keys())
    
    for i, (col, metric) in enumerate(zip(cols, metric_names[:3])):
        with col:
            st.metric(
                label=metric.replace("_", " ").title(),
                value=f"{scores[metric]:.1f}/10"
            )
    
    if len(metric_names) > 3:
        cols2 = st.columns(3)
        for i, (col, metric) in enumerate(zip(cols2, metric_names[3:6])):
            with col:
                st.metric(
                    label=metric.replace("_", " ").title(),
                    value=f"{scores[metric]:.1f}/10"
                )

