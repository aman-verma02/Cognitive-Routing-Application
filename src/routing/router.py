import numpy as np
from typing import List, Dict, Any

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError("Please install langchain-huggingface, langchain-community, and faiss-cpu") from e

from ..utils.logger import get_logger
from ..utils.exceptions import RoutingError

logger = get_logger(__name__)

# Define the provided bot personas
BOT_PERSONAS = [
    {
        "id": "Bot A",
        "description": "Tech Maximalist",
        "persona": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
    },
    {
        "id": "Bot B",
        "description": "Doomer / Skeptic",
        "persona": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature."
    },
    {
        "id": "Bot C",
        "description": "Finance Bro",
        "persona": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
    }
]

class PersonaRouter:
    def __init__(self):
        logger.info("Initializing Vector Store for Persona Router.")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = self._initialize_vector_store()
            logger.info("Vector Store initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {str(e)}")
            raise RoutingError(f"Vector Store Initialization failed: {str(e)}") from e

    def _initialize_vector_store(self) -> FAISS:
        """Embeds the bot personas and stores them in an in-memory FAISS vector database."""
        logger.debug(f"Embedding {len(BOT_PERSONAS)} personas.")
        documents = [
            Document(page_content=bot["persona"], metadata={"id": bot["id"], "description": bot["description"]})
            for bot in BOT_PERSONAS
        ]
        
        # Create a FAISS vector store from the documents
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store

    def route_post_to_bots(self, post_content: str, threshold: float = 0.40) -> List[Dict[str, Any]]:
        """
        Takes a post, embeds it, and queries the vector store to find bots that care about it.
        Uses cosine similarity. If similarity > threshold, the bot is returned.
        
        Note: The threshold for realistic results depends on the embedding model.
        Using a smaller model like all-MiniLM-L6-v2 typically yields higher base similarity scores.
        We'll use Euclidean distance as returned by FAISS by default and convert it or 
        just use similarity search with score.
        """
        # FAISS similarity_search_with_relevance_scores typically returns scores normalized between 0 and 1
        # where 1 is highest similarity. Some models might need tweaking.
        
        logger.info(f"Routing post to bots (Threshold: {threshold})")
        logger.debug(f"Post content: {post_content}")
        
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(post_content, k=3)
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise RoutingError(f"Similarity search failed: {str(e)}") from e
        
        interested_bots = []
        for doc, score in results:
            if score >= threshold:
                logger.debug(f"Bot matched: {doc.metadata['id']} with score {score:.4f}")
                interested_bots.append({
                    "bot_id": doc.metadata["id"],
                    "description": doc.metadata["description"],
                    "persona": doc.page_content,
                    "similarity_score": float(score)
                })
        
        logger.info(f"Found {len(interested_bots)} interested bots.")
        return interested_bots
