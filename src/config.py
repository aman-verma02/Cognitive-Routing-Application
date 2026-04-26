import os
from dotenv import load_dotenv
from .utils.logger import get_logger
from .utils.exceptions import LLMConfigurationError

logger = get_logger(__name__)

# Load environment variables from .env if present
load_dotenv()
logger.info("Loaded environment variables.")

# We can fall back to a mock/fake LLM if no API keys are provided for testing purposes.
# But normally we'd expect them to be set.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not OPENAI_API_KEY and not GROQ_API_KEY:
    logger.warning("No API keys found in .env. LLM calls in Phase 2 and 3 will fail!")

def get_llm():
    """
    Helper to get the appropriate LLM instance based on available env variables.
    Defaults to ChatOpenAI (which can be used for OpenAI, Ollama, or vLLM).
    Falls back to Groq if specified.
    """
    logger.debug("Initializing LLM.")
    try:
        if GROQ_API_KEY:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama3-8b-8192", temperature=0.7)
        else:
            # Default to OpenAI compatible endpoint
            from langchain_openai import ChatOpenAI
            # Using a default model. For OpenAI it might be gpt-4o-mini or gpt-3.5-turbo.
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=OPENAI_API_KEY or "dummy",
                base_url=OPENAI_API_BASE
            )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMConfigurationError(f"Error configuring LLM: {str(e)}") from e
