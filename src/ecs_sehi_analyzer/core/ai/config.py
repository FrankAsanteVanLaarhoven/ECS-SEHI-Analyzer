import os
from dotenv import load_dotenv

def load_ai_config():
    """Load AI configuration from environment variables"""
    load_dotenv()
    
    config = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "org_id": os.getenv("OPENAI_ORG_ID"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4")
        },
        "perplexity": {
            "api_key": os.getenv("PERPLEXITY_API_KEY"),
            "base_url": os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
            "model": os.getenv("PERPLEXITY_MODEL", "pplx-70b-chat")
        }
    }
    
    return config 