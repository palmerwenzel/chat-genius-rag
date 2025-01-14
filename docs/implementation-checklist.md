# Implementation Checklist: RAG and Chatbots (MVP)

## 1. RAG Repo (Python / LangChain / LangSmith / Pinecone)

1. **Repository Structure** ✅
   - [x] Set up Python project with FastAPI
     - Implemented in `main.py` using FastAPI framework
     - Basic app structure with CORS middleware and environment validation
   - [x] Configure environment variables
     - Environment template in `.env.sample`
     - Required variables: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
   - [x] Implement basic health check endpoint
     - `/health` endpoint in `main.py`
     - Returns service status and version

2. **Chatbot Seeding Service** ✅
   - [x] Implement endpoint for generating AI conversations
     - `/api/seed` endpoint in `main.py`
     - Accepts prompt and number of turns
   - [x] Create two distinct bot personalities
     - Defined in `chatbots.py` as TechBot and PMBot
     - Each with unique characteristics and communication styles
   - [x] Handle conversation flow and message generation
     - `ChatbotConversation` class in `chatbots.py`
     - Alternates between personalities for natural dialogue
   - [x] Return formatted messages for storage
     - Messages include role, content, and metadata
     - Compatible with Supabase schema

3. **RAG Pipeline Implementation** ✅
   - [x] Set up document indexing using Pinecone
     - `RAGPipeline` class in `rag.py`
     - Uses OpenAI embeddings and Pinecone vector store
   - [x] Implement similarity search
     - `search_similar` method in `rag.py`
     - Based on reference implementation from `similarity_search.ipynb`
   - [x] Create summarization pipeline
     - `generate_summary` method in `rag.py`
     - Uses context-aware prompting with similar messages
   - [x] Expose summary endpoint
     - `/api/summary` endpoint in `main.py`
     - Accepts messages and optional query parameter

4. **LangSmith Integration** ✅
   - [x] Configure LangSmith for debugging
     - `LangSmithLogger` class in `langsmith_logger.py`
     - Integrated with LangChain's tracing system
   - [x] Add basic request/response logging
     - Decorator-based tracing with `@langsmith_logger.trace_chain`
     - Logs chain execution, inputs, outputs, and errors

5. **Deployment**
   - [x] Deploy Python service to AWS EC2
     - Containerized with `Dockerfile` and `docker-compose.yml`
     - Deployment script in `deploy.sh`
   - [x] Set up basic monitoring
     - Monitoring script in `monitor.sh`
     - Health checks, resource monitoring, and auto-restart
   - [x] Configure CORS for Next.js app communication
     - CORS middleware in `main.py`
     - Currently allows all origins (TODO: restrict in production)

## Next Steps

1. Start with the Python RAG service:
   - [x] Implement basic FastAPI structure
   - [x] Add chatbot seeding functionality
   - [x] Test RAG pipeline locally

2. Update Next.js app:
   - [ ] Add @bot commands
   - [ ] Implement API routes
   - [ ] Test integration with EC2 endpoint

3. Deploy and validate:
   - [x] Deploy Python service to EC2
   - [ ] Test end-to-end flow
   - [ ] Generate seed data
   - [ ] Verify summary functionality

Notes:
- MVP focuses on basic functionality over optimization
- Using existing message schema (no database changes needed)
- Simple EC2 deployment for Python service
- Command-based interaction only (@bot commands) 