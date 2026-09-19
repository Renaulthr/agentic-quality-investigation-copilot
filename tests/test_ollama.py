from src.llm.ollama_llm import OllamaLLM
import pytest

pytestmark = pytest.mark.integration

llm = OllamaLLM()

response = llm.generate(
    "Reply only with: Python Ollama connection successful"
)

print(response)