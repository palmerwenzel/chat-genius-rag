from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import json
import logging
import os

# Configure logging for errors only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Define bot IDs
BOT_IDS = {
    "SYSTEM": "00000000-0000-0000-0000-000000000000",
    "BOT_1": "00000000-0000-0000-0000-000000000b01",
    "BOT_2": "00000000-0000-0000-0000-000000000b02",
    "BOT_3": "00000000-0000-0000-0000-000000000b03",
    "BOT_4": "00000000-0000-0000-0000-000000000b04",
    "BOT_5": "00000000-0000-0000-0000-000000000b05",
    "BOT_6": "00000000-0000-0000-0000-000000000b06",
    "BOT_7": "00000000-0000-0000-0000-000000000b07",
    "BOT_8": "00000000-0000-0000-0000-000000000b08",
    "BOT_9": "00000000-0000-0000-0000-000000000b09",
    "BOT_10": "00000000-0000-0000-0000-000000000b10"
}

# Define bot personalities and roles
BOT_PERSONAS = {
    BOT_IDS["SYSTEM"]: {
        "name": "System",
        "role": "System coordinator and administrator",
        "persona": "You are System Bot, a clear and authoritative coordinator who manages the chat system.",
        "temperature": 0.3
    },
    BOT_IDS["BOT_1"]: {
        "name": "TechBot",
        "role": "Technical Expert",
        "persona": "You are TechBot, a precise technical expert who explains software concepts with detailed examples.",
        "temperature": 0.7
    },
    BOT_IDS["BOT_2"]: {
        "name": "ProductBot",
        "role": "Product Manager",
        "persona": "You are ProductBot, a strategic product manager who focuses on user value and market fit.",
        "temperature": 0.7
    },
    BOT_IDS["BOT_3"]: {
        "name": "DesignBot",
        "role": "UX/UI Designer",
        "persona": "You are DesignBot, a UX/UI designer who creates intuitive and accessible experiences.",
        "temperature": 0.8
    },
    BOT_IDS["BOT_4"]: {
        "name": "DataBot",
        "role": "Data Scientist",
        "persona": "You are DataBot, an analytical data scientist who backs statements with evidence.",
        "temperature": 0.6
    },
    BOT_IDS["BOT_5"]: {
        "name": "SecurityBot",
        "role": "Security Expert",
        "persona": "You are SecurityBot, a security expert who emphasizes best practices and threat prevention.",
        "temperature": 0.5
    },
    BOT_IDS["BOT_6"]: {
        "name": "DevOpsBot",
        "role": "DevOps Engineer",
        "persona": "You are DevOpsBot, a practical DevOps engineer focused on automation and reliability.",
        "temperature": 0.6
    },
    BOT_IDS["BOT_7"]: {
        "name": "TestBot",
        "role": "QA Engineer",
        "persona": "You are TestBot, a thorough QA engineer who considers edge cases and user scenarios.",
        "temperature": 0.6
    },
    BOT_IDS["BOT_8"]: {
        "name": "AgileBot",
        "role": "Agile Coach",
        "persona": "You are AgileBot, an agile coach who improves team efficiency and processes.",
        "temperature": 0.7
    },
    BOT_IDS["BOT_9"]: {
        "name": "ArchitectBot",
        "role": "Software Architect",
        "persona": "You are ArchitectBot, a strategic architect who designs scalable systems.",
        "temperature": 0.6
    },
    BOT_IDS["BOT_10"]: {
        "name": "AccessibilityBot",
        "role": "Accessibility Expert",
        "persona": "You are AccessibilityBot, an advocate for inclusive design and WCAG compliance.",
        "temperature": 0.7
    }
}

class ChatbotConversation:
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the chatbot conversation handler.
        
        Args:
            model_name (str): The name of the LLM model to use
        """
        self.llm = ChatOpenAI(model_name=model_name)
        self.bot_personas = BOT_PERSONAS
        
    def get_bot_id_by_name(self, bot_name: str) -> Optional[str]:
        """Get a bot's ID by its name."""
        for bot_id, persona in self.bot_personas.items():
            if persona["name"].lower() == bot_name.lower():
                return bot_id
        return None

    def list_available_bots(self) -> List[Dict[str, str]]:
        """Return a list of available bots with their names and roles."""
        return [
            {
                "id": bot_id,
                "name": info["name"],
                "role": info["role"]
            }
            for bot_id, info in self.bot_personas.items()
        ]

    def get_bot_persona(self, bot_id: str) -> Dict[str, Any]:
        """Get the persona details for a specific bot."""
        return self.bot_personas.get(bot_id, {})

    def get_all_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get all bot personas."""
        return self.bot_personas

    def generate_response(
        self,
        bot_id: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate a response from a specific bot."""
        if not self.validate_bot_id(bot_id):
            raise ValueError(f"Invalid bot ID: {bot_id}. Available bots: {', '.join(self.list_available_bots())}")

        bot_info = self.bot_personas[bot_id]
        temp = temperature if temperature is not None else bot_info.get("temperature", 0.7)
        
        # Create a new LLM instance with the bot's temperature
        llm = ChatOpenAI(temperature=temp, model_name=self.llm.model_name)

        system_message = {
            "role": "system",
            "content": f"""You are {bot_info['name']}, {bot_info['role']}. {bot_info['persona']}
            Respond in character, maintaining your unique perspective and expertise."""
        }

        try:
            response = llm.invoke([system_message, *messages])
            
            return {
                "role": "assistant",
                "content": response.content,
                "metadata": {
                    "bot_id": bot_id,
                    "bot_name": bot_info["name"],
                    "bot_role": bot_info["role"]
                }
            }
        except Exception as e:
            logger.error(f"Error generating response for bot {bot_id}: {str(e)}")
            raise

    def generate_conversation(
        self,
        initial_prompt: str,
        num_turns: int = 3,
        bot_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate a multi-turn conversation between bots.
        
        Args:
            initial_prompt: The prompt to start the conversation
            num_turns: Number of conversation turns
            bot_ids: List of bot IDs to participate. If None, randomly selects from available bots
        """
        if not bot_ids:
            # Default to TechBot and ProductBot if no bots specified
            bot_ids = [BOT_IDS["BOT_1"], BOT_IDS["BOT_2"]]
        
        conversation = [{
            "role": "user",
            "content": initial_prompt
        }]
        
        for i in range(num_turns):
            bot_id = bot_ids[i % len(bot_ids)]
            response = self.generate_response(
                bot_id=bot_id,
                messages=conversation
            )
            conversation.append(response)
            
        return conversation

    def validate_bot_id(self, bot_id: str) -> bool:
        """Validate if a bot ID exists."""
        return bot_id in self.bot_personas 

    def update_bot_persona(self, bot_id: str, persona: str) -> None:
        """Update a bot's persona."""
        if not self.validate_bot_id(bot_id):
            raise ValueError(f"Invalid bot ID: {bot_id}")
        
        self.bot_personas[bot_id]["persona"] = persona 