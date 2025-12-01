"""
Research Display Component for Streamlit UI.
"""

import streamlit as st
from typing import Dict, Any
import plotly.graph_objects as go


def render_research_results(result: Dict[str, Any]):
    """
    Render research results in Streamlit.
    
    Args:
        result: Research result dictionary from orchestrator
    """
    if not result:
        st.warning("No research results to display.")
        return
    
    # Display report
    st.subheader("Research Report")
    st.markdown(result.get("report", "No report generated."))
    
    # Display quality scores
    st.subheader("Quality Evaluation")
    quality_scores = result.get("quality_scores", {})
    if quality_scores:
        # Create radar chart
        categories = list(quality_scores.keys())
        values = [quality_scores[k] for k in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=[c.replace("_", " ").title() for c in categories] + [categories[0].replace("_", " ").title()],
            fill='toself',
            name='Quality Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            title="Quality Scores"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display average score
        avg_score = result.get("average_quality_score", 0)
        st.metric("Overall Quality Score", f"{avg_score:.1f}/10")
    
    # Display sources
    st.subheader("Sources")
    sources = result.get("sources", [])
    if sources:
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source.get('title', 'Unknown')}"):
                if source.get("type") == "web":
                    st.write(f"**URL:** {source.get('url', 'N/A')}")
                elif source.get("type") == "document":
                    st.write(f"**Document:** {source.get('document_id', 'N/A')}")
                    if source.get("page"):
                        st.write(f"**Page:** {source.get('page')}")
    else:
        st.info("No sources available.")
    
    # Display citations
    st.subheader("Citations")
    citations = result.get("citations", [])
    if citations:
        citation_format = st.selectbox(
            "Citation Format",
            ["APA", "MLA", "Chicago"],
            key="citation_format"
        )
        
        from backend.core.citation_manager import CitationManager
        citation_manager = CitationManager()
        citation_manager.citations = citations
        
        formatted_citations = citation_manager.format_citations(
            format_type=citation_format.lower()
        )
        
        for i, citation in enumerate(formatted_citations, 1):
            st.write(f"{i}. {citation}")
    else:
        st.info("No citations available.")

