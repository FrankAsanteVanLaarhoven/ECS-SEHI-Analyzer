import pytest
from src.ecs_sehi_analyzer.core.sandbox import SandboxInterface

def test_sandbox_initialization():
    sandbox = SandboxInterface()
    assert sandbox is not None
    assert "python" in sandbox.supported_languages 