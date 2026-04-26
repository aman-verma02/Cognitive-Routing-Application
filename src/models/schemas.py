from pydantic import BaseModel, Field

class GeneratedPost(BaseModel):
    """Schema for the final JSON output of a bot's drafted post."""
    bot_id: str = Field(description="The identifier or name of the bot creating the post")
    topic: str = Field(description="The topic the post is about")
    post_content: str = Field(description="The drafted content of the post, max 280 characters")
