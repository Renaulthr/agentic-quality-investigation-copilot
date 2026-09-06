from src.rag.context_builder import (
    build_context,
)

from src.rag.prompt_builder import (
    build_rag_prompt,
)

from src.llm.ollama_llm import OllamaLLM


class RAGService:

    def __init__(
        self,
        top_k: int = 3,
    ):

        self.top_k = top_k

        self.llm = OllamaLLM(model="llama3.2")


    def answer(
        self,
        question: str,
    ):

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        context, sources = (
            build_context(
                query=question,
                top_k=self.top_k,
            )
        )

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        answer = self.llm.generate(
            prompt
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

if __name__ == "__main__":

    rag = RAGService()

    result = rag.answer(
        "What controls are defined "
        "for detecting a missing circlip?"
    )

    print(
        "\nANSWER\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nRETRIEVED SOURCES\n"
    )

    for source in result[
        "sources"
    ]:

        print(
            source
        )    