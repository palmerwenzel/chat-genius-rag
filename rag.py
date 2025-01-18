from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from pinecone import Pinecone as PineconeClient
import os
import logging
import time

# Configure logging for errors only
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4"):
        logger.info("Initializing RAG with Pinecone")
        # Initialize Pinecone
        self.pc = PineconeClient(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")  # Get from env var instead of hardcoding
        )
        logger.info("Pinecone client initialized")
        
        # Initialize other components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("OpenAI embeddings initialized")
        self.llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        self.index_name = os.getenv("PINECONE_INDEX")
        
        # Ensure index exists and get it
        self._ensure_index()
        
        # Initialize vector store with explicit namespace
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace=os.getenv("PINECONE_NAMESPACE", "default")  # Use configured namespace or default
        )
        logger.info(f"Connected to Pinecone index: {self.index_name} with namespace: {os.getenv('PINECONE_NAMESPACE', 'default')}")

    def _ensure_index(self):
        """Ensure the Pinecone index exists and is properly configured."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new index: {self.index_name}")
                # Create index with appropriate configuration
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # Dimension for text-embedding-3-small
                    metric="cosine"
                )
                logger.info(f"Successfully created index: {self.index_name}")
            
            # Get the index
            self.index = self.pc.Index(self.index_name)
            
            # Initialize the index with a default namespace if needed
            try:
                stats = self.index.describe_index_stats()
                logger.info(f"Index stats: {stats}")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
            
            logger.info(f"Successfully connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error ensuring index exists: {str(e)}", exc_info=True)
            raise

    async def reset_index(self) -> None:
        """Reset the vector store by deleting all vectors."""
        try:
            logger.info("Starting index reset")
            
            # Get index statistics
            stats = self.index.describe_index_stats()
            logger.info(f"Current index stats before reset: {stats}")
            
            # Try to delete all vectors without specifying namespace
            try:
                self.index.delete(delete_all=True)
                logger.info("Successfully deleted all vectors")
            except Exception as e:
                logger.warning(f"Error during global delete: {e}")
                # If global delete fails, try namespace-specific delete
                if stats.get('namespaces'):
                    for namespace in stats['namespaces'].keys():
                        try:
                            self.index.delete(delete_all=True, namespace=namespace)
                            logger.info(f"Deleted vectors from namespace: {namespace}")
                        except Exception as ns_error:
                            logger.error(f"Error deleting namespace {namespace}: {ns_error}")
            
            # Verify deletion
            after_stats = self.index.describe_index_stats()
            logger.info(f"Index stats after reset: {after_stats}")
            
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}", exc_info=True)
            raise

    def index_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Index chat messages into Pinecone with comprehensive metadata."""
        try:
            logger.info(f"Processing {len(messages)} messages for indexing")
            
            documents = []
            for msg in messages:
                logger.info(f"Converting message from {msg['role']} to document")
                if msg["role"] in ["assistant", "human"]:
                    metadata = {
                        "role": msg["role"],
                        "channel_id": msg.get("channel_id", "default"),
                        "group_id": msg.get("group_id", "default"),
                        "sender_id": msg.get("sender_id", "unknown"),
                        "sender_name": msg.get("metadata", {}).get("sender_name", "unknown"),
                        "timestamp": msg.get("created_at", ""),
                        **{k: str(v) for k, v in msg.get("metadata", {}).items()}
                    }
                    
                    metadata = {k: str(v) for k, v in metadata.items() if v is not None}
                    
                    doc = Document(
                        page_content=msg["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
            
            if not documents:
                logger.warning("No valid documents to index")
                return

            logger.info("Splitting documents into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} chunks")
            
            # Get stats before indexing
            before_stats = self.index.describe_index_stats()
            logger.info(f"Index stats before indexing: {before_stats}")
            
            # Calculate expected count before indexing
            expected_count = before_stats.get('total_vector_count', 0) + len(split_docs)
            
            # Add to vectorstore with explicit namespace
            logger.info("Generating embeddings and storing in Pinecone")
            self.vectorstore.add_documents(split_docs)
            
            # Wait for indexing to complete with retries
            max_retries = 5  # Increased retries
            for i in range(max_retries):
                time.sleep(10)  # Increased wait time between checks
                after_stats = self.index.describe_index_stats()
                actual_count = after_stats.get('total_vector_count', 0)
                if actual_count >= expected_count:
                    logger.info(f"Vectors successfully indexed after {i+1} retries")
                    break
                logger.info(f"Waiting for indexing to complete... ({i+1}/{max_retries})")
            else:
                logger.error(f"Indexing did not complete after {max_retries} retries")
                raise Exception("Indexing timeout")

            logger.info("Successfully indexed all messages")
        except Exception as e:
            logger.error(f"Error indexing messages: {str(e)}", exc_info=True)
            raise

    def search_similar(
        self, 
        query: str, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Document]:
        """Search for similar messages with flexible metadata filtering."""
        try:
            logger.info(f"Searching for documents similar to query: {query}")
            
            # Reduced wait time for indexing to settle
            time.sleep(1)
            
            search_kwargs = {"k": k}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata
                
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            results = retriever.invoke(query)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error searching similar messages: {str(e)}", exc_info=True)
            raise
    
    def generate_summary(
        self, 
        messages: List[Dict[str, Any]], 
        query: str = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a summary with flexible context filtering."""
        try:
            logger.info("Starting summary generation")
            logger.info(f"Using filters: {filter_metadata}")
            
            # Get relevant context if query is provided
            context = ""
            if query:
                similar_docs = self.search_similar(
                    query, 
                    filter_metadata=filter_metadata
                )
                context = "\n".join(doc.page_content for doc in similar_docs)
            
            # Create summary prompt with metadata context
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that generates concise but informative 
                 summaries of chat conversations. Focus on key points, decisions, and outcomes."""),
                ("system", f"Additional context: {context}" if context else ""),
                ("user", """Please summarize the following chat conversation. Focus on the main topics 
                 discussed and key takeaways.\n\nConversation:\n{conversation}""")
            ])
            
            # Format conversation for summary
            conversation_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            ])
            
            # Generate summary
            logger.info("Generating summary with ChatOpenAI")
            summary_response = self.llm.invoke(
                summary_prompt.format(conversation=conversation_text)
            )
            logger.info("Summary generated successfully")
            
            return summary_response.content
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            raise 