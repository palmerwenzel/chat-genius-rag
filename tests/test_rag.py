import pytest
import httpx
import os
import sys
from pathlib import Path
import time
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag import RAGPipeline
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Test data
TEST_MESSAGES = [
    {
        "role": "human",
        "content": "How do I implement authentication in Next.js?",
        "metadata": {
            "sender_name": "TestUser",
            "timestamp": "2024-01-18T12:00:00Z"
        }
    },
    {
        "role": "assistant",
        "content": "For Next.js authentication, you can use NextAuth.js. Here's a basic implementation...",
        "metadata": {
            "bot_name": "TechBot",
            "bot_role": "Technical Expert"
        }
    }
]

@pytest.fixture
def rag_pipeline():
    """Initialize RAG pipeline for testing."""
    return RAGPipeline(model_name="gpt-3.5-turbo")

@pytest.fixture
def api_client():
    """Initialize API client for testing."""
    return httpx.AsyncClient(
        base_url="http://localhost:8000",
        headers={
            "Authorization": f"Bearer {os.getenv('RAG_SERVICE_API_KEY')}"
        }
    )

async def wait_for_index_update(
    index,
    expected_count: int,
    max_retries: int = 20,
    delay: int = 3,
    namespace: Optional[str] = None
) -> bool:
    """Wait for index to reach expected count."""
    for i in range(max_retries):
        stats = index.describe_index_stats()
        current_count = (
            stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
            if namespace
            else stats.get('total_vector_count', 0)
        )
        
        if current_count >= expected_count:
            logger.info(f"Index reached expected count after {i+1} retries")
            return True
            
        logger.info(f"Waiting for index update... Current: {current_count}, Expected: {expected_count} (Attempt {i+1}/{max_retries})")
        await asyncio.sleep(delay)
    
    return False

async def test_direct_indexing(rag_pipeline):
    """Test indexing directly through the RAG pipeline."""
    logger.info("Starting direct indexing test")
    
    # Index test messages
    rag_pipeline.index_messages(TEST_MESSAGES)
    
    # Wait for indexing to complete with longer total wait time
    success = await wait_for_index_update(
        rag_pipeline.index,
        expected_count=2,
        max_retries=20,  # Up to 60 seconds total wait (20 × 3)
        delay=3
    )
    assert success, "Indexing did not complete within expected time"
    
    # Search for similar content to verify indexing
    results = rag_pipeline.search_similar(
        query="authentication Next.js",
        k=1
    )
    
    # Detailed assertion messages
    assert len(results) > 0, "No results found after indexing"
    assert "authentication" in results[0].page_content.lower(), f"Expected 'authentication' in content but got: {results[0].page_content}"
    
    # Verify final index state
    final_stats = rag_pipeline.index.describe_index_stats()
    logger.info(f"Final index stats: {final_stats}")
    assert final_stats['total_vector_count'] > 0, "No vectors found in final index check"
    
    # Clean up: reset the index
    await rag_pipeline.reset_index()
    
    # Verify cleanup
    cleanup_stats = rag_pipeline.index.describe_index_stats()
    logger.info(f"Cleanup index stats: {cleanup_stats}")

async def test_api_indexing(api_client):
    """Test indexing through the API endpoint."""
    logger.info("Starting API indexing test")
    
    # Index messages through API
    index_response = await api_client.post("/api/index", json={"messages": TEST_MESSAGES})
    assert index_response.status_code == 200
    
    # Wait for indexing to complete with retries
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    success = await wait_for_index_update(
        index,
        expected_count=2,
        max_retries=20,
        delay=3
    )
    assert success, "API indexing did not complete within expected time"
    
    # Verify through a summary request
    summary_response = await api_client.post(
        "/api/summary",
        json={
            "messages": TEST_MESSAGES,
            "query": "authentication Next.js"
        }
    )
    assert summary_response.status_code == 200
    summary_data = summary_response.json()
    assert "authentication" in summary_data["summary"].lower(), f"Expected 'authentication' in summary but got: {summary_data['summary']}"
    
    # Clean up: reset the index
    reset_response = await api_client.post("/api/index/reset")
    assert reset_response.status_code == 200

if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        # Initialize fixtures
        rag = RAGPipeline(model_name="gpt-3.5-turbo")
        async with httpx.AsyncClient(
            base_url="http://localhost:8000",
            headers={
                "Authorization": f"Bearer {os.getenv('RAG_SERVICE_API_KEY')}"
            }
        ) as client:
            # Run tests
            print("Testing direct indexing...")
            await test_direct_indexing(rag)
            print("Direct indexing test passed!")
            
            print("\nTesting API indexing...")
            await test_api_indexing(client)
            print("API indexing test passed!")

    asyncio.run(run_tests()) 