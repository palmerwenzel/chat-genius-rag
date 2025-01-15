# Chat Genius RAG Test Commands

## Bot Commands
Test these commands in the chat interface:

```bash
# Seed a conversation between bots
@bot seed Tell me about modern web development frameworks

# Generate channel summary
@bot summary
@bot summary focusing on technical discussions

# View current bot personas
@bot personas

# Set custom bot personas
@bot set-personas --bot1 "a creative storyteller who uses vivid analogies" --bot2 "a methodical analyst who breaks down complex topics"

# Reset Pinecone index (required after changing personas)
@bot reset-index

# Index channel messages
@bot index
```

## Testing with Payloads
Use these commands to test the RAG service directly:

```bash
# Test all endpoints
cd chat-genius-rag
./test_endpoints.sh

# Test individual endpoints
curl -X GET http://localhost:8000/api/personas \
  -H "Authorization: Bearer ${RAG_SERVICE_API_KEY}"

curl -X POST http://localhost:8000/api/personas \
  -H "Authorization: Bearer ${RAG_SERVICE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @test_payloads/personas_set.json

curl -X POST http://localhost:8000/api/index/reset \
  -H "Authorization: Bearer ${RAG_SERVICE_API_KEY}"

curl -X POST http://localhost:8000/api/index \
  -H "Authorization: Bearer ${RAG_SERVICE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @test_payloads/index_messages.json

curl -X POST http://localhost:8000/api/summary \
  -H "Authorization: Bearer ${RAG_SERVICE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @test_payloads/summary_query.json
```

## Important Notes
1. Always reset the index after changing personas to ensure consistency
2. The RAG service must be running (`docker-compose up`) before testing
3. Set environment variables in `.env` before testing:
   ```bash
   export RAG_SERVICE_API_KEY=your_api_key
   ```
4. Test payloads are in `test_payloads/` directory
5. Monitor the RAG service logs for detailed debugging information

## Expected Behavior
1. Persona changes should be reflected in subsequent bot conversations
2. Index reset should clear all vectors from Pinecone
3. Summaries should reflect the current personas and indexed content
4. All endpoints should require valid API key authentication 