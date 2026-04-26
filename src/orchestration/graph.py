import json
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from ..models.schemas import GeneratedPost
from .tools import mock_searxng_search
from ..config import get_llm
from ..utils.logger import get_logger
from ..utils.exceptions import OrchestrationError

logger = get_logger(__name__)

# Define the State for LangGraph
class BotState(TypedDict):
    bot_id: str
    bot_persona: str
    search_query: str
    search_results: str
    post_draft: GeneratedPost

# Nodes
def decide_search(state: BotState):
    """Node 1: Decide Search. The LLM decides what topic it wants to post about and formats a search query."""
    logger.info(f"Node: decide_search started for bot_id={state.get('bot_id', 'unknown')}")
    try:
        llm = get_llm()
        persona = state["bot_persona"]
        
        prompt = f"""
        You are an autonomous bot with the following persona:
        "{persona}"
        
        Based on your persona, what topic do you want to post about today?
        Generate a simple, 2-4 word search query to look up recent news about this topic.
        Return ONLY the search query string, nothing else.
        """
        
        response = llm.invoke([SystemMessage(content=prompt)])
        query = response.content.strip().strip('"\'')
        logger.debug(f"Search query decided: {query}")
        
        return {"search_query": query}
    except Exception as e:
        logger.error(f"Error in decide_search node: {str(e)}")
        raise OrchestrationError(f"decide_search failed: {str(e)}") from e

def web_search(state: BotState):
    """Node 2: Web Search. Executes the mock_searxng_search tool to get real-world context."""
    logger.info("Node: web_search started")
    try:
        query = state["search_query"]
        
        # We call our mock tool
        results = mock_searxng_search.invoke({"query": query})
        logger.debug(f"Web search results retrieved length: {len(results)}")
        
        return {"search_results": results}
    except Exception as e:
        logger.error(f"Error in web_search node: {str(e)}")
        raise OrchestrationError(f"web_search failed: {str(e)}") from e

def draft_post(state: BotState):
    """Node 3: Draft Post. The LLM uses Persona + Search Results to generate a JSON post."""
    logger.info("Node: draft_post started")
    try:
        llm = get_llm()
        # We want a structured output (JSON object)
        structured_llm = llm.with_structured_output(GeneratedPost)
        
        persona = state["bot_persona"]
        results = state["search_results"]
        bot_id = state["bot_id"]
        
        prompt = f"""
        You are an autonomous bot with the following persona:
        "{persona}"
        
        You recently searched the web and found this news context:
        "{results}"
        
        Draft a highly opinionated post (max 280 characters) responding to or commenting on this news.
        The post must heavily reflect your persona. 
        Output a structured JSON with your bot_id "{bot_id}", the core topic, and the post_content.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        logger.debug("Draft post completed successfully.")
        
        return {"post_draft": response}
    except Exception as e:
        logger.error(f"Error in draft_post node: {str(e)}")
        raise OrchestrationError(f"draft_post failed: {str(e)}") from e

# Build the Graph
def build_autonomous_content_engine():
    workflow = StateGraph(BotState)
    
    workflow.add_node("decide_search", decide_search)
    workflow.add_node("web_search", web_search)
    workflow.add_node("draft_post", draft_post)
    
    workflow.add_edge(START, "decide_search")
    workflow.add_edge("decide_search", "web_search")
    workflow.add_edge("web_search", "draft_post")
    workflow.add_edge("draft_post", END)
    
    return workflow.compile()

# Helper to run the graph easily
def run_autonomous_post_generation(bot_id: str, bot_persona: str) -> GeneratedPost:
    logger.info(f"Starting autonomous post generation for bot_id={bot_id}")
    try:
        engine = build_autonomous_content_engine()
        initial_state = {
            "bot_id": bot_id,
            "bot_persona": bot_persona,
            "search_query": "",
            "search_results": "",
            "post_draft": None
        }
        
        final_state = engine.invoke(initial_state)
        logger.info(f"Autonomous post generation completed for bot_id={bot_id}")
        return final_state["post_draft"]
    except Exception as e:
        logger.error(f"Failed to run autonomous generation: {str(e)}")
        raise OrchestrationError(f"Graph execution failed: {str(e)}") from e
