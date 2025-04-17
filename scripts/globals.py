# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
from collections import deque
import os
import operator
from typing import Union, List, Tuple

# Third-party imports
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langgraph.store.memory import InMemoryStore


# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
# Load environment variables from .env file
_ = load_dotenv()

# ==============================================================================
# CONVERSATION HISTORY
# ==============================================================================
# Variables to store conversation history with limited capacity
global last_human_messages
last_human_messages = deque(maxlen=3)  # Limit to 3 messages

global last_agent_messages
last_agent_messages = deque(maxlen=3)  # Limit to 3 responses

# ==============================================================================
# APPLICATION CONFIGURATION
# ==============================================================================
# Initial settings and available locations
global initial_location
initial_location = "init"

global config
config = {"configurable": {"langgraph_user_id": "david"}, "recursion_limit": 25}

global places
places = ["init","bedroom","gym","dinner_table","kitchen","sofa"]

# ==============================================================================
# EMBEDDING AND STORAGE SETUP
# ==============================================================================
# Initialize embedding model with Azure OpenAI credentials
global embed_model
embed_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# In-memory vector store for embeddings
global store
store = InMemoryStore(index={"embed": embed_model})


# ==============================================================================
# DATA MODELS
# ==============================================================================
# Type definition for plan execution state
class PlanExecute(TypedDict):
    """State container for the plan-execute workflow."""
    input: str  # User input
    plan: List[str]  # Current plan steps
    past_steps: Annotated[List[Tuple], operator.add]  # Accumulated executed steps
    response: str  # Final response to user


class Plan(BaseModel):
    """Plan model defining a sequence of steps to follow."""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response model for sending messages back to the user."""
    response: str


class Act(BaseModel):
    """Action model for determining the next system action."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )