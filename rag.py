from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any
from langsmith_logger import langsmith_logger
import os

class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        self.index_name = os.getenv("PINECONE_INDEX")
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
    
    @langsmith_logger.trace_chain("rag_indexing")
    def index_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Index chat messages into Pinecone."""
        # Convert messages to documents
        documents = []
        for msg in messages:
            # Only index assistant messages and human messages
            if msg["role"] in ["assistant", "human"]:
                doc = Document(
                    page_content=msg["content"],
                    metadata={
                        "role": msg["role"],
                        **msg.get("metadata", {})
                    }
                )
                documents.append(doc)
        
        # Split documents into chunks if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Add to vectorstore
        self.vectorstore.add_documents(split_docs)
    
    @langsmith_logger.trace_chain("rag_search")
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar messages using the query."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    
    @langsmith_logger.trace_chain("rag_summarize")
    def generate_summary(self, messages: List[Dict[str, Any]], query: str = None) -> str:
        """Generate a summary of the chat messages."""
        # First, get relevant context if query is provided
        context = ""
        if query:
            similar_docs = self.search_similar(query)
            context = "\n".join(doc.page_content for doc in similar_docs)
        
        # Create summary prompt
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
        summary_response = self.llm.invoke(
            summary_prompt.format(conversation=conversation_text)
        )
        
        return summary_response.content 