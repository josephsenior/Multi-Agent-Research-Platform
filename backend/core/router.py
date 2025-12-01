"""
Router for the Multi-Agent Research Platform.

Determines research strategy and coordinates agent workflow using routing pattern.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()


class Router:
    """
    Router that determines research strategy and coordinates agent workflow.
    
    Uses routing pattern to:
    - Analyze research queries
    - Determine optimal research strategy
    - Coordinate agent workflow
    - Handle different query types
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Router.
        
        Args:
            model_name: OpenAI model name
            temperature: LLM temperature
            api_key: OpenAI API key
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
        
        self._build_router()
    
    def _build_router(self):
        """Build the routing chain."""
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research strategy coordinator. Analyze research queries 
and determine the optimal research strategy.

Consider:
1. Query complexity (simple fact vs. comprehensive analysis)
2. Query type (factual, analytical, comparative, etc.)
3. Information needs (current events, historical, technical, etc.)
4. Required depth (quick answer vs. deep dive)

Determine:
- Whether to use web search, document retrieval, or both
- How many sources to gather
- What level of fact-checking is needed
- Whether the query requires specialized handling

Respond with a JSON-like structure indicating the strategy."""),
            ("user", "Research query: {query}")
        ])
        
        self.router_chain = router_prompt | self.llm | StrOutputParser()
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a research query and determine strategy.
        
        Args:
            query: Research query
            
        Returns:
            Dictionary with routing decision and strategy
        """
        routing_result = self.router_chain.invoke({"query": query})
        
        # Parse routing decision
        strategy = self._parse_strategy(routing_result, query)
        
        return {
            "query": query,
            "strategy": strategy,
            "routing_text": routing_result
        }
    
    def _parse_strategy(self, routing_text: str, query: str) -> Dict[str, Any]:
        """
        Parse routing text to extract strategy.
        
        Args:
            routing_text: Raw routing output
            query: Original query
            
        Returns:
            Strategy dictionary
        """
        # Default strategy
        strategy = {
            "use_web_search": True,
            "use_rag": True,
            "max_web_results": 5,
            "max_rag_results": 5,
            "fact_check_level": "standard",
            "complexity": "medium",
            "query_type": "general"
        }
        
        routing_lower = routing_text.lower()
        
        # Determine complexity
        if any(word in routing_lower for word in ["simple", "quick", "factual", "basic"]):
            strategy["complexity"] = "simple"
            strategy["max_web_results"] = 3
            strategy["max_rag_results"] = 3
            strategy["fact_check_level"] = "basic"
        elif any(word in routing_lower for word in ["complex", "comprehensive", "deep", "detailed"]):
            strategy["complexity"] = "complex"
            strategy["max_web_results"] = 8
            strategy["max_rag_results"] = 8
            strategy["fact_check_level"] = "thorough"
        
        # Determine query type
        if any(word in routing_lower for word in ["compare", "comparison", "versus", "vs"]):
            strategy["query_type"] = "comparative"
        elif any(word in routing_lower for word in ["how", "why", "explain", "analyze"]):
            strategy["query_type"] = "analytical"
        elif any(word in routing_lower for word in ["what", "when", "where", "who"]):
            strategy["query_type"] = "factual"
        
        # Determine if RAG should be used
        if any(word in routing_lower for word in ["document", "uploaded", "file", "pdf"]):
            strategy["use_rag"] = True
        elif any(word in routing_lower for word in ["current", "recent", "latest", "news"]):
            strategy["use_rag"] = False  # Prefer web for current events
        
        return strategy

