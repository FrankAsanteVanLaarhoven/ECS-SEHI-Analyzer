import openai
import httpx
import asyncio
from typing import Dict, Optional
from .config import load_ai_config

class AIEngine:
    def __init__(self):
        self.config = load_ai_config()
        self._setup_openai()
        self._setup_perplexity()
        
    def _setup_openai(self):
        """Configure OpenAI client"""
        openai.api_key = self.config["openai"]["api_key"]
        if self.config["openai"]["org_id"]:
            openai.organization = self.config["openai"]["org_id"]
            
    def _setup_perplexity(self):
        """Configure Perplexity client"""
        self.perplexity_client = httpx.AsyncClient(
            base_url=self.config["perplexity"]["base_url"],
            headers={
                "Authorization": f"Bearer {self.config['perplexity']['api_key']}",
                "Content-Type": "application/json"
            }
        )
        
    async def query_openai(self, prompt: str, context: str = "") -> Dict:
        """Query OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ]
            )
            return response
        except Exception as e:
            return {"error": str(e)}
            
    async def query_perplexity(self, prompt: str, context: str = "") -> Dict:
        """Query Perplexity API"""
        try:
            response = await self.perplexity_client.post(
                "/chat/completions",
                json={
                    "model": self.config["perplexity"]["model"],
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API keys are configured"""
        return {
            "openai": bool(self.config["openai"]["api_key"]),
            "perplexity": bool(self.config["perplexity"]["api_key"])
        } 