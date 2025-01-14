# Chat Genius RAG Service

A RAG (Retrieval Augmented Generation) service that provides AI chatbot functionality and conversation summarization capabilities.

## Features

- **AI Chatbot Conversations**
  - Two distinct AI personalities: TechBot and PMBot
  - Natural dialogue generation with alternating perspectives
  - Configurable conversation length
  - Structured message output with metadata

- **RAG Pipeline**
  - Document indexing with Pinecone vector store
  - Semantic similarity search
  - Context-aware conversation summarization
  - Query-based retrieval

- **Monitoring & Debugging**
  - LangSmith integration for chain tracing
  - Request/response logging
  - Health checks and resource monitoring
  - Auto-restart capabilities

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API Keys:
  - OpenAI API key
  - Pinecone API key and index
  - LangChain API key

## Quick Start

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd chat-genius-rag
   ```

2. Set up environment variables:
   ```bash
   cp .env.sample .env
   # Edit .env with your API keys
   ```

3. Run with Docker:
   ```bash
   docker-compose up --build
   ```

The service will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```http
GET /health
```

### Generate AI Conversation
```http
POST /api/seed
{
    "prompt": "string",
    "num_turns": integer (default: 3)
}
```

### Index Messages
```http
POST /api/index
{
    "messages": [
        {
            "role": "string",
            "content": "string",
            "metadata": object
        }
    ]
}
```

### Generate Summary
```http
POST /api/summary
{
    "messages": [
        {
            "role": "string",
            "content": "string",
            "metadata": object
        }
    ],
    "query": "string" (optional)
}
```

## Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

## Deployment

1. Configure AWS EC2 instance with Docker and Docker Compose

2. Deploy the service:
   ```bash
   chmod +x deploy.sh monitor.sh
   ./deploy.sh
   ```

3. Start monitoring:
   ```bash
   ./monitor.sh
   ```

## Project Structure

```
chat-genius-rag/
├── main.py              # FastAPI application
├── chatbots.py          # Chatbot conversation handling
├── rag.py              # RAG pipeline implementation
├── langsmith_logger.py  # LangSmith integration
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
├── deploy.sh           # Deployment script
├── monitor.sh          # Monitoring script
├── Dockerfile          # Container definition
├── docker-compose.yml  # Container orchestration
└── docs/              # Documentation
    └── implementation-checklist.md
```

## Documentation

- API documentation available at `/docs` or `/redoc`
- Implementation details in `docs/implementation-checklist.md`

## Monitoring

The monitoring script (`monitor.sh`) provides:
- Health checks
- Container status monitoring
- System resource tracking
- Automatic container restart

## Contributing

1. Check the implementation checklist
2. Follow the code style guidelines
3. Submit pull requests with tests

## License

[License Type] - see LICENSE file for details 