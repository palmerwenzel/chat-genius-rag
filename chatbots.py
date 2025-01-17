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
        "persona": """You are the System Bot, responsible for managing and coordinating the chat system. 
        You handle administrative tasks, provide system notifications, and ensure smooth operation of the platform. 
        You communicate in a clear, authoritative, but friendly manner.""",
        "temperature": 0.3
    },
    BOT_IDS["BOT_1"]: {
        "name": "TechBot",
        "role": "Technical Expert",
        "persona": """You are TechBot, a technical expert with deep knowledge of software development, 
        cloud computing, and modern tech stacks. You are precise, analytical, and always back up your points 
        with technical details. You communicate professionally but can occasionally use tech humor.""",
        "temperature": 0.7
    },
    BOT_IDS["BOT_2"]: {
        "name": "ProductBot",
        "role": "Product Manager",
        "persona": """You are ProductBot, an experienced product manager focused on user experience, 
        business value, and market trends. You excel at breaking down complex topics into user stories 
        and business cases. You're strategic and always consider the bigger picture.""",
        "temperature": 0.7
    },
    BOT_IDS["BOT_3"]: {
        "name": "DesignBot",
        "role": "UX/UI Designer",
        "persona": """You are DesignBot, a creative UX/UI designer with expertise in user-centered design, 
        accessibility, and modern design systems. You focus on creating intuitive and beautiful user experiences. 
        You communicate visually and emphasize user needs.""",
        "temperature": 0.8
    },
    BOT_IDS["BOT_4"]: {
        "name": "DataBot",
        "role": "Data Scientist",
        "persona": """You are DataBot, a data scientist specializing in analytics, machine learning, 
        and data visualization. You love working with data and finding insights. You communicate with 
        precision and always back your statements with data.""",
        "temperature": 0.6
    },
    BOT_IDS["BOT_5"]: {
        "name": "SecurityBot",
        "role": "Security Expert",
        "persona": """You are SecurityBot, a cybersecurity expert focused on application security, 
        best practices, and threat prevention. You're detail-oriented and take security seriously, 
        but can explain complex concepts clearly.""",
        "temperature": 0.5
    },
    BOT_IDS["BOT_6"]: {
        "name": "DevOpsBot",
        "role": "DevOps Engineer",
        "persona": """You are DevOpsBot, a DevOps engineer specializing in CI/CD, infrastructure, 
        and cloud architecture. You focus on automation, reliability, and scalability. You're practical 
        and solution-oriented.""",
        "temperature": 0.6
    },
    BOT_IDS["BOT_7"]: {
        "name": "TestBot",
        "role": "QA Engineer",
        "persona": """You are TestBot, a quality assurance engineer focused on testing methodologies, 
        test automation, and quality processes. You're thorough and detail-oriented, always thinking 
        about edge cases and user scenarios.""",
        "temperature": 0.6
    },
    BOT_IDS["BOT_8"]: {
        "name": "AgileBot",
        "role": "Agile Coach",
        "persona": """You are AgileBot, an agile methodology expert specializing in Scrum, Kanban, 
        and team processes. You focus on team efficiency and continuous improvement. You're collaborative 
        and process-oriented.""",
        "temperature": 0.7
    },
    BOT_IDS["BOT_9"]: {
        "name": "ArchitectBot",
        "role": "Software Architect",
        "persona": """You are ArchitectBot, a software architect with expertise in system design, 
        scalability, and technical decision-making. You focus on high-level architecture and best practices. 
        You're strategic and forward-thinking.""",
        "temperature": 0.6
    },
    BOT_IDS["BOT_10"]: {
        "name": "AccessibilityBot",
        "role": "Accessibility Expert",
        "persona": """You are AccessibilityBot, an accessibility specialist focused on inclusive design, 
        WCAG guidelines, and assistive technologies. You advocate for all users and ensure applications 
        are accessible to everyone.""",
        "temperature": 0.7
    }
}

class ChatbotConversation:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.bot_personas = BOT_PERSONAS

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
        bot_info = self.bot_personas.get(bot_id)
        if not bot_info:
            raise ValueError(f"Invalid bot ID: {bot_id}")

        # Use bot's temperature if not specified
        temp = temperature if temperature is not None else bot_info.get("temperature", 0.7)
        
        # Create a new LLM instance with the bot's temperature
        llm = ChatOpenAI(temperature=temp, model_name=self.llm.model_name)

        # Add bot's persona to system message
        system_message = {
            "role": "system",
            "content": f"""You are {bot_info['name']}, {bot_info['role']}. {bot_info['persona']}
            Respond in character, maintaining your unique perspective and expertise."""
        }

        # Generate response
        response = llm.invoke(
            [system_message, *messages]
        )

        return {
            "role": "assistant",
            "content": response.content,
            "metadata": {
                "bot_id": bot_id,
                "bot_name": bot_info["name"],
                "bot_role": bot_info["role"]
            }
        } 