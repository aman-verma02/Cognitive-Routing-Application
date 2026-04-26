class CognitiveRoutingError(Exception):
    """Base exception class for all custom errors in Cognitive Routing system."""
    pass

class LLMConfigurationError(CognitiveRoutingError):
    """Raised when there is an issue with LLM API keys or configuration."""
    pass

class RoutingError(CognitiveRoutingError):
    """Raised when an error occurs during Vector-based persona matching (Phase 1)."""
    pass

class OrchestrationError(CognitiveRoutingError):
    """Raised when an error occurs during LangGraph execution (Phase 2)."""
    pass

class CombatEngineError(CognitiveRoutingError):
    """Raised when an error occurs during RAG prompt generation or execution (Phase 3)."""
    pass
