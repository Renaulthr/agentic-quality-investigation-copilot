import pytest

from src.llm.ollama_llm import OllamaLLM


pytestmark = pytest.mark.integration


def test_ollama_generation():
    """
    Integration test for the local Ollama service.

    Requires:
    - Ollama server running
    - llama3.2 model installed
    - OLLAMA_BASE_URL configured if not using localhost
    """

    llm = OllamaLLM(
        model="llama3.2",
    )

    response = llm.generate(
        "Reply exactly with: Ollama working"
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0