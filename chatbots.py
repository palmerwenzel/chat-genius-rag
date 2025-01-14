from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from typing import List, Dict, Any
from langsmith_logger import langsmith_logger
import json

# Define bot personalities
PERSONALITIES = {
    "tech_expert": """You are TechBot, a technical expert with deep knowledge of software development, 
    cloud computing, and modern tech stacks. You are precise, analytical, and always back up your points 
    with technical details. You communicate professionally but can occasionally use tech humor.""",
    
    "product_manager": """You are PMBot, an experienced product manager focused on user experience, 
    business value, and market trends. You excel at breaking down complex topics into user stories 
    and business cases. You're strategic and always consider the bigger picture."""
}

class ChatbotConversation:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=model_name
        )
        self.conversation_history: List[Dict[str, Any]] = []
    
    @langsmith_logger.trace_chain("chatbot_prompt")
    def _create_bot_prompt(self, personality: str, context: str, history: List[Dict[str, Any]]) -> str:
        """Create a prompt for the bot based on personality and context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", PERSONALITIES[personality]),
            ("system", f"Context for this conversation: {context}"),
            *[("assistant" if msg["role"] == "assistant" else "human", msg["content"]) 
              for msg in history],
        ])
        return prompt
    
    @langsmith_logger.trace_chain("chatbot_conversation")
    def generate_conversation(self, 
                            initial_prompt: str, 
                            num_turns: int = 3) -> List[Dict[str, Any]]:
        """Generate a conversation between two AI chatbots."""
        self.conversation_history = []
        current_speaker = "tech_expert"
        
        # Add the initial prompt
        self.conversation_history.append({
            "role": "human",
            "content": initial_prompt,
            "metadata": {"type": "user_prompt"}
        })
        
        for _ in range(num_turns):
            # Create prompt for current bot
            prompt = self._create_bot_prompt(
                personality=current_speaker,
                context=initial_prompt,
                history=self.conversation_history
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
                "metadata": {
                    "type": "bot_message",
                    "personality": current_speaker
                }
            })
            
            # Switch speakers
            current_speaker = "product_manager" if current_speaker == "tech_expert" else "tech_expert"
        
        return self.conversation_history 