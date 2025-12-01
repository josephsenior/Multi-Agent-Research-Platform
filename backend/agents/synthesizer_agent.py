"""
Synthesizer Agent for the Multi-Agent Research Platform.

Organizes and structures research findings into coherent reports:
- Organize findings by topic
- Create coherent narrative
- Generate structured output
- Include citations
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent


class SynthesizerAgent(BaseAgent):
    """
    Agent specialized in organizing and structuring information.
    
    Capabilities:
    - Organize findings by topic
    - Create coherent narrative
    - Generate structured output
    - Include proper citations
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Synthesizer Agent.
        
        Args:
            model_name: OpenAI model name
            temperature: LLM temperature
            api_key: OpenAI API key
        """
        system_prompt = """You are a professional research synthesizer. Your role is to 
organize and structure research findings into coherent, well-formatted reports.

When synthesizing:
1. Organize information by topic/themes
2. Create a logical flow and narrative
3. Use clear headings and structure
4. Include citations for all claims
5. Maintain accuracy while improving readability
6. Highlight key findings and insights

Create professional, publication-ready reports."""
        
        super().__init__(
            role="Synthesizer",
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
    
    def process(
        self,
        verified_findings: Dict[str, Any],
        query: str,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesize verified findings into a structured report.
        
        Args:
            verified_findings: Verified findings from Fact-Checker Agent
            query: Original research query
            include_citations: Whether to include citations
            
        Returns:
            Dictionary with synthesized report and metadata
        """
        # Prepare synthesis prompt
        synthesis_input = f"""Original Research Query: {query}

Research Findings:
{verified_findings.get('findings', {}).get('synthesized_findings', '')}

Verification Report:
{verified_findings.get('verification_report', '')}

Confidence Score: {verified_findings.get('confidence_score', 0)}/10

Please synthesize this information into a comprehensive, well-structured research report.

The report should include:
1. Executive Summary (brief overview)
2. Main Findings (organized by topic)
3. Key Insights
4. Limitations and Uncertainties (if any)
5. Conclusion

Format the report with clear headings, bullet points where appropriate, and proper 
citations using [Source X] notation. Make it professional and easy to read."""
        
        synthesized_report = self._invoke(synthesis_input)
        
        # Post-process to add structure
        structured_report = self._add_structure(synthesized_report, query)
        
        return {
            "report": structured_report,
            "query": query,
            "confidence_score": verified_findings.get("confidence_score", 0),
            "sources_count": verified_findings.get("sources_checked", 0),
            "word_count": len(structured_report.split())
        }
    
    def _add_structure(self, report: str, query: str) -> str:
        """
        Add structure and formatting to the report.
        
        Args:
            report: Raw synthesized report
            query: Original query
            
        Returns:
            Structured report with proper formatting
        """
        # Add title if not present
        if not report.startswith("#"):
            structured = f"# Research Report: {query}\n\n"
            structured += f"**Query:** {query}\n\n"
            structured += "---\n\n"
            structured += report
        else:
            structured = report
        
        return structured
    
    def create_summary(
        self,
        report: str,
        max_length: int = 200
    ) -> str:
        """
        Create a brief summary of the report.
        
        Args:
            report: Full research report
            max_length: Maximum summary length
            
        Returns:
            Brief summary
        """
        summary_input = f"""Research Report:
{report[:2000]}...

Create a brief summary ({max_length} words or less) that captures the key findings 
and conclusions."""
        
        return self._invoke(summary_input)

