from src.llm.ollama_llm import OllamaLLM


llm = OllamaLLM()

response = llm.generate(
    "Reply only with: Python Ollama connection successful"
)

print(response)