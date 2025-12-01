"""
Researcher Agent for the Multi-Agent Research Platform.

Gathers information from multiple sources:
- Web search (Tavily/Serper API)
- Document retrieval via RAG
- Multi-source information gathering
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from ..core.rag_system import RAGSystem
from ..tools.web_search import WebSearchTool
from ..core.citation_manager import CitationManager


class ResearcherAgent(BaseAgent):
    """
    Agent specialized in gathering information from multiple sources.
    
    Capabilities:
    - Web search via Tavily/Serper API
    - Document retrieval via RAG
    - Multi-source information gathering
    - Source tracking and citation extraction
    """
    
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        web_search_tool: Optional[WebSearchTool] = None,
        citation_manager: Optional[CitationManager] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Researcher Agent.
        
        Args:
            rag_system: RAG system for document retrieval
            web_search_tool: Web search tool
            citation_manager: Citation manager for tracking sources
            model_name: OpenAI model name
            temperature: LLM temperature
            api_key: OpenAI API key
        """
        system_prompt = """You are a research specialist. Your role is to gather 
comprehensive information on topics from multiple sources. 

When researching:
1. Use web search to find current information
2. Use document retrieval to find relevant information from uploaded documents
3. Synthesize information from both sources
4. Track all sources for citation
5. Provide detailed, well-sourced information

Always cite your sources and indicate the reliability of information."""
        
        super().__init__(
            role="Researcher",
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.rag_system = rag_system
        self.web_search_tool = web_search_tool
        self.citation_manager = citation_manager
    
    def process(
        self,
        query: str,
        use_web_search: bool = True,
        use_rag: bool = True,
        max_web_results: int = 5,
        max_rag_results: int = 5
    ) -> Dict[str, Any]:
        """
        Conduct research on a topic.
        
        Args:
            query: Research query
            use_web_search: Whether to use web search
            use_rag: Whether to use RAG document retrieval
            max_web_results: Maximum web search results
            max_rag_results: Maximum RAG retrieval results
            
        Returns:
            Dictionary with research findings, sources, and citations
        """
        findings = {
            "query": query,
            "web_results": [],
            "rag_results": [],
            "synthesized_findings": "",
            "sources": [],
            "citation_ids": []
        }
        
        # Web search
        if use_web_search and self.web_search_tool:
            try:
                web_results = self.web_search_tool.search(query, max_results=max_web_results)
                findings["web_results"] = web_results.get("results", [])
                
                # Add citations
                for result in web_results.get("sources", []):
                    if self.citation_manager:
                        citation_id = self.citation_manager.add_web_citation(
                            url=result.get("url", ""),
                            title=result.get("title", ""),
                            content_snippet=result.get("content", "")[:200] if result.get("content") else None
                        )
                        findings["citation_ids"].append(citation_id)
                        findings["sources"].append({
                            "type": "web",
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "citation_id": citation_id
                        })
            except Exception as e:
                print(f"[Researcher] Web search error: {e}")
                findings["web_search_error"] = str(e)
                # Continue without web search if it fails
                if not findings.get("rag_results"):
                    raise RuntimeError(f"Both web search and RAG failed. Web search error: {str(e)}")
        
        # RAG document retrieval
        if use_rag and self.rag_system:
            try:
                rag_results = self.rag_system.query(query, k=max_rag_results)
                findings["rag_results"] = rag_results.get("retrieved_documents", [])
                
                # Add citations
                for source in rag_results.get("sources", []):
                    if self.citation_manager:
                        citation_id = self.citation_manager.add_document_citation(
                            document_id=source.get("document_id", "unknown"),
                            page=source.get("page"),
                            title=source.get("source", "Document"),
                            content_snippet=source.get("content", "")[:200] if source.get("content") else None
                        )
                        findings["citation_ids"].append(citation_id)
                        findings["sources"].append({
                            "type": "document",
                            "document_id": source.get("document_id"),
                            "page": source.get("page"),
                            "citation_id": citation_id
                        })
            except Exception as e:
                print(f"[Researcher] RAG retrieval error: {e}")
                findings["rag_error"] = str(e)
                # Continue without RAG if it fails
                if not findings.get("web_results"):
                    raise RuntimeError(f"Both RAG and web search failed. RAG error: {str(e)}")
        
        # Synthesize findings
        findings["synthesized_findings"] = self._synthesize_findings(
            query,
            findings["web_results"],
            findings["rag_results"]
        )
        
        return findings
    
    def _synthesize_findings(
        self,
        query: str,
        web_results: List[Dict[str, Any]],
        rag_results: List[str]
    ) -> str:
        """
        Synthesize findings from multiple sources.
        
        Args:
            query: Original query
            web_results: Web search results
            rag_results: RAG retrieval results
            
        Returns:
            Synthesized research findings
        """
        # Prepare context
        web_context = ""
        if web_results:
            web_context = "Web Search Results:\n"
            for i, result in enumerate(web_results, 1):
                web_context += f"\n[{i}] {result.get('title', 'No title')}\n"
                web_context += f"URL: {result.get('url', 'No URL')}\n"
                web_context += f"Content: {result.get('content', 'No content')[:300]}...\n"
        
        rag_context = ""
        if rag_results:
            rag_context = "\n\nDocument Retrieval Results:\n"
            for i, result in enumerate(rag_results, 1):
                rag_context += f"\n[{i}] {result[:300]}...\n"
        
        # Create synthesis prompt
        synthesis_input = f"""Research Query: {query}

{web_context}
{rag_context}

Based on the information from web search and document retrieval above, synthesize 
a comprehensive research finding. Include:
1. Key information relevant to the query
2. Important details from multiple sources
3. Any contradictions or uncertainties
4. Source references

Provide a well-structured synthesis that combines information from all sources."""
        
        return self._invoke(synthesis_input)

