from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import json
import logging
import os

# Configure logging for errors only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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
    def __init__(
        self, 
        model_name: str = "gpt-4",
        bot1_persona: str = "a technical expert who explains concepts clearly and thoroughly",
        bot2_persona: str = "a practical problem-solver who focuses on real-world applications"
    ):
        self.llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        self.bot1_persona = bot1_persona
        self.bot2_persona = bot2_persona

    def set_personas(self, bot1_persona: str, bot2_persona: str) -> None:
        """Update the personas for both bots."""
        self.bot1_persona = bot1_persona
        self.bot2_persona = bot2_persona

    def get_personas(self) -> Dict[str, str]:
        """Get the current personas for both bots."""
        return {
            "bot1": self.bot1_persona,
            "bot2": self.bot2_persona
        }

    def generate_conversation(
        self,
        initial_prompt: str,
        num_turns: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate a conversation between two AI chatbots."""
        
        system_prompt = f"""You are participating in a conversation between two AI chatbots discussing: "{initial_prompt}"

Bot 1 is {self.bot1_persona}.
Bot 2 is {self.bot2_persona}.

Each response should be a single message from one bot to the other, naturally continuing the conversation.
Keep responses concise (2-3 paragraphs max) and engaging. Use a casual, friendly tone while maintaining expertise."""

        messages = [{"role": "system", "content": system_prompt}]

        # Generate conversation turns
        for i in range(num_turns * 2):
            current_bot = "Bot 1" if i % 2 == 0 else "Bot 2"
            current_persona = self.bot1_persona if i % 2 == 0 else self.bot2_persona
            
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in messages[1:]],
                    {
                        "role": "system",
                        "content": f"You are {current_bot} ({current_persona}). Respond to continue the conversation."
                    }
                ]
            )

            content = response.content if hasattr(response, 'content') else str(response)
            messages.append({
                "role": "assistant",
                "content": content,
                "metadata": {
                    "bot_number": (i % 2) + 1,
                    "persona": current_persona
                }
            })

        # Return only the bot messages
        return [m for m in messages if m["role"] == "assistant"] 