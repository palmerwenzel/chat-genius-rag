from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from chatbots import ChatbotConversation
from rag import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Chat Genius RAG",
    description="RAG and AI Chatbot Service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "RAG_SERVICE_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize services with default personas
DEFAULT_BOT1_PERSONA = "a technical expert who explains concepts clearly and thoroughly"
DEFAULT_BOT2_PERSONA = "a practical problem-solver who focuses on real-world applications"

chatbot = ChatbotConversation(
    model_name="gpt-3.5-turbo",
    bot1_persona=os.getenv("BOT1_PERSONA", DEFAULT_BOT1_PERSONA),
    bot2_persona=os.getenv("BOT2_PERSONA", DEFAULT_BOT2_PERSONA)
)

# Request/Response models
class SeedConversationRequest(BaseModel):
    prompt: str
    num_turns: int = 3
    bot1_persona: Optional[str] = None
    bot2_persona: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )

class SeedConversationResponse(BaseModel):
    messages: List[Message]

class IndexMessagesRequest(BaseModel):
    messages: List[Message]

class GenerateSummaryRequest(BaseModel):
    messages: List[Message]
    query: Optional[str] = None

class GenerateSummaryResponse(BaseModel):
    summary: str

class SetPersonasRequest(BaseModel):
    bot1_persona: str
    bot2_persona: str

class SetPersonasResponse(BaseModel):
    message: str
    bot1_persona: str
    bot2_persona: str

class ResetIndexResponse(BaseModel):
    status: str
    message: str

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="No API key provided"
        )
    
    expected_api_key = os.getenv("RAG_SERVICE_API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=500, detail="RAG_SERVICE_API_KEY not configured"
        )
    
    if api_key != f"Bearer {expected_api_key}":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    
    return api_key

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# Chatbot seeding endpoint
@app.post("/api/seed", response_model=SeedConversationResponse)
async def seed_conversation(request: SeedConversationRequest):
    try:
        conversation = chatbot.generate_conversation(
            initial_prompt=request.prompt,
            num_turns=request.num_turns,
            bot1_persona=request.bot1_persona,
            bot2_persona=request.bot2_persona
        )
        messages = [Message.from_dict(msg) for msg in conversation]
        return SeedConversationResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index messages endpoint
@app.post("/api/index")
async def index_messages(
    request: IndexMessagesRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        rag.index_messages([msg.dict() for msg in request.messages])
        return {"status": "success", "message": "Messages indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate summary endpoint
@app.post("/api/summary", response_model=GenerateSummaryResponse)
async def generate_summary(
    request: GenerateSummaryRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        summary = rag.generate_summary(
            messages=[msg.dict() for msg in request.messages],
            query=request.query
        )
        return GenerateSummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/personas", response_model=SetPersonasResponse)
async def set_personas(
    request: SetPersonasRequest,
    api_key: str = Depends(verify_api_key)
):
    """Set the personas for the two chatbots."""
    try:
        chatbot.set_personas(
            bot1_persona=request.bot1_persona,
            bot2_persona=request.bot2_persona
        )
        return SetPersonasResponse(
            message="Bot personas updated successfully",
            bot1_persona=request.bot1_persona,
            bot2_persona=request.bot2_persona
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/personas", response_model=SetPersonasResponse)
async def get_personas(api_key: str = Depends(verify_api_key)):
    """Get the current personas for the two chatbots."""
    try:
        personas = chatbot.get_personas()
        return SetPersonasResponse(
            message="Current bot personas",
            bot1_persona=personas["bot1"],
            bot2_persona=personas["bot2"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/reset", response_model=ResetIndexResponse)
async def reset_index(api_key: str = Depends(verify_api_key)):
    """Reset the vector store by deleting all indexed messages."""
    try:
        await rag.reset_index()
        return ResetIndexResponse(
            status="success",
            message="Successfully reset index"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 