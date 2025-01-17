from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from pinecone import Pinecone as PineconeClient
import os
import logging

# Configure logging for errors only
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4"):
        logger.info("Initializing RAG with Pinecone")
        # Initialize Pinecone
        self.pc = PineconeClient(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        logger.info("Pinecone client initialized")
        
        # Initialize other components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("OpenAI embeddings initialized")
        self.llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        self.index_name = os.getenv("PINECONE_INDEX")
        
        # Get Pinecone index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def index_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Index chat messages into Pinecone with comprehensive metadata."""
        try:
            logger.info(f"Processing {len(messages)} messages for indexing")
            
            documents = []
            for msg in messages:
                logger.info(f"Converting message from {msg['role']} to document")
                if msg["role"] in ["assistant", "human"]:
                    # Extract and validate crucial metadata
                    metadata = {
                        "role": msg["role"],
                        "channel_id": msg.get("channel_id"),
                        "group_id": msg.get("group_id"),
                        "sender_id": msg.get("sender_id"),
                        "sender_name": msg.get("metadata", {}).get("sender_name"),
                        "timestamp": msg.get("created_at"),
                        **msg.get("metadata", {})  # Preserve any additional metadata
                    }
                    
                    # Remove None values to keep metadata clean
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    
                    doc = Document(
                        page_content=msg["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
            
            # Split documents into chunks if needed
            logger.info("Splitting documents into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} chunks")
            
            # Add to vectorstore
            logger.info("Generating embeddings and storing in Pinecone")
            self.vectorstore.add_documents(split_docs)
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
            logger.info(f"Applying filters: {filter_metadata}")
            
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
    
    async def reset_index(self) -> None:
        """Reset the vector store by deleting all vectors."""
        try:
            logger.info("Starting index reset")
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            logger.info("Successfully reset index")
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}", exc_info=True)
            raise 