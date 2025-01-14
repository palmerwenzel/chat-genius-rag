from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    "LANGCHAIN_PROJECT"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize services
chatbot = ChatbotConversation(model_name="gpt-4")
rag = RAGPipeline(model_name="gpt-4")

# Request/Response models
class SeedConversationRequest(BaseModel):
    prompt: str
    num_turns: int = 3

class Message(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any]

class SeedConversationResponse(BaseModel):
    messages: List[Message]

class IndexMessagesRequest(BaseModel):
    messages: List[Message]

class GenerateSummaryRequest(BaseModel):
    messages: List[Message]
    query: Optional[str] = None

class GenerateSummaryResponse(BaseModel):
    summary: str

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
            num_turns=request.num_turns
        )
        return SeedConversationResponse(messages=conversation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index messages endpoint
@app.post("/api/index")
async def index_messages(request: IndexMessagesRequest):
    try:
        rag.index_messages([msg.dict() for msg in request.messages])
        return {"status": "success", "message": "Messages indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate summary endpoint
@app.post("/api/summary", response_model=GenerateSummaryResponse)
async def generate_summary(request: GenerateSummaryRequest):
    try:
        summary = rag.generate_summary(
            messages=[msg.dict() for msg in request.messages],
            query=request.query
        )
        return GenerateSummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 