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

chatbot = ChatbotConversation(model_name="gpt-4")

# Initialize RAG pipeline
rag = RAGPipeline(model_name="gpt-3.5-turbo")

# Request/Response models
class SeedConversationRequest(BaseModel):
    prompt: str
    num_turns: int = 3
    bot_ids: Optional[List[str]] = None  # Optional list of bot UUIDs to participate

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

class ChatRequest(BaseModel):
    bot_id: str
    messages: List[Message]
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    message: Message

class ListBotsResponse(BaseModel):
    bots: List[Dict[str, str]]

class BotPersona(BaseModel):
    """Model for a single bot's persona details"""
    name: str
    role: str
    persona: str
    temperature: float

class GetPersonasResponse(BaseModel):
    """Response model for getting bot personas"""
    personas: Dict[str, BotPersona]

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
async def seed_conversation(
    request: SeedConversationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate a seeded conversation between bots."""
    try:
        # Validate bot IDs if provided
        if request.bot_ids:
            for bot_id in request.bot_ids:
                if not chatbot.validate_bot_id(bot_id):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid bot ID: {bot_id}"
                    )

        conversation = chatbot.generate_conversation(
            initial_prompt=request.prompt,
            num_turns=request.num_turns,
            bot_ids=request.bot_ids
        )
        messages = [Message.from_dict(msg) for msg in conversation]
        return SeedConversationResponse(messages=messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

@app.get("/api/bots", response_model=ListBotsResponse)
async def list_bots(api_key: str = Depends(verify_api_key)):
    """Get a list of all available bots."""
    try:
        bots = chatbot.list_available_bots()
        return ListBotsResponse(bots=bots)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def generate_chat_response(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate a response from a specific bot."""
    try:
        response = chatbot.generate_response(
            bot_id=request.bot_id,
            messages=[msg.dict() for msg in request.messages],
            temperature=request.temperature
        )
        return ChatResponse(message=Message.from_dict(response))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

@app.get("/api/personas", response_model=GetPersonasResponse)
async def get_personas(
    bot_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Get bot personas."""
    try:
        if bot_id:
            # Get specific bot's persona
            if not chatbot.validate_bot_id(bot_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bot ID: {bot_id}"
                )
            info = chatbot.get_bot_persona(bot_id)
            personas = {
                bot_id: BotPersona(
                    name=info["name"],
                    role=info["role"],
                    persona=info["persona"],
                    temperature=info["temperature"]
                )
            }
        else:
            # Get all personas
            personas = {
                bot_id: BotPersona(
                    name=info["name"],
                    role=info["role"],
                    persona=info["persona"],
                    temperature=info["temperature"]
                )
                for bot_id, info in chatbot.get_all_personas().items()
            }
        
        return GetPersonasResponse(personas=personas)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class UpdatePersonaRequest(BaseModel):
    """Request model for updating bot personas"""
    personas: Dict[str, str]

class UpdatePersonaResponse(BaseModel):
    """Response model for updating bot personas"""
    message: str
    personas: Dict[str, str]

@app.post("/api/personas", response_model=UpdatePersonaResponse)
async def update_personas(
    request: UpdatePersonaRequest,
    api_key: str = Depends(verify_api_key)
):
    """Update multiple bot personas at once."""
    try:
        updated_personas = {}
        for key, persona in request.personas.items():
            # Extract bot ID from key (e.g., "bot1_persona" -> "1")
            bot_id = key.replace("bot", "").replace("_persona", "")
            
            if not chatbot.validate_bot_id(bot_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bot ID in key: {key}"
                )
            
            # Update the persona
            bot_info = chatbot.get_bot_persona(bot_id)
            bot_info["persona"] = persona
            updated_personas[key] = persona
        
        return UpdatePersonaResponse(
            message="Successfully updated bot personas",
            personas=updated_personas
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 