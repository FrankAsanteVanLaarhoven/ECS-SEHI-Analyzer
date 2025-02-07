import pytest
from src.ecs_sehi_analyzer.core.ai.engine import AIEngine

@pytest.mark.asyncio
async def test_ai_engine():
    engine = AIEngine()
    
    # Test API key validation
    validation = engine.validate_api_keys()
    assert validation["openai"] is True, "OpenAI API key not configured"
    assert validation["perplexity"] is True, "Perplexity API key not configured"
    
    # Test OpenAI query
    response = await engine.query_openai(
        "Explain quantum entanglement",
        "Scientific explanation context"
    )
    assert "error" not in response, "OpenAI query failed"
    
    # Test Perplexity query
    response = await engine.query_perplexity(
        "Explain quantum entanglement",
        "Scientific explanation context"
    )
    assert "error" not in response, "Perplexity query failed" 