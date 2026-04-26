from langchain_core.tools import tool
from ..utils.logger import get_logger

logger = get_logger(__name__)

@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a web search using a mock SearXNG endpoint.
    Given a search query, returns hardcoded, recent news headlines.
    """
    query_lower = query.lower()
    logger.info(f"Executing mock search for query: '{query}'")
    
    # Return mock results based on keyword matching
    if "crypto" in query_lower or "bitcoin" in query_lower:
        logger.debug("Matched 'crypto' keyword.")
        return "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum follows suit."
    elif "ai" in query_lower or "openai" in query_lower or "model" in query_lower:
        logger.debug("Matched 'ai' keyword.")
        return "OpenAI just released a new model that might replace junior developers. Tech sector reacts with mixed feelings."
    elif "elon" in query_lower or "musk" in query_lower or "space" in query_lower:
        logger.debug("Matched 'elon/space' keyword.")
        return "SpaceX successfully lands Starship prototype. Elon Musk tweets about Mars colony timeline."
    elif "market" in query_lower or "rate" in query_lower or "economy" in query_lower:
        logger.debug("Matched 'market' keyword.")
        return "Federal Reserve hints at cutting interest rates in the upcoming quarter. Markets rally."
    elif "ev" in query_lower or "battery" in query_lower or "electric" in query_lower:
        logger.debug("Matched 'ev' keyword.")
        return "New solid-state EV batteries show promising results, retaining 95% capacity after extensive testing."
    else:
        logger.debug("No specific keywords matched, returning general news.")
        return "General news: Global markets remain steady as tech innovations continue to drive growth."
