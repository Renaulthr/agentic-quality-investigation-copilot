import time

import mlflow
from mlflow.entities import SpanType
from ollama import Client

from src.llm.base import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(
            host=host
        )

    @mlflow.trace(
        name="ollama_generate",
        span_type=SpanType.LLM,
    )
    def generate(
        self,
        prompt: str,
    ) -> str:

        start_time = time.perf_counter()

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
                "seed": 42,
            },
        )

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        answer = response[
            "message"
        ][
            "content"
        ]

        prompt_tokens = response.get(
            "prompt_eval_count",
            0,
        )

        completion_tokens = response.get(
            "eval_count",
            0,
        )

        total_tokens = (
            prompt_tokens
            + completion_tokens
        )

        span = (
            mlflow.get_current_active_span()
        )

        if span is not None:

            span.set_attributes(
                {
                    "llm.provider":
                        "ollama",

                    "llm.model":
                        self.model,

                    "llm.latency_seconds":
                        round(
                            latency_seconds,
                            4,
                        ),

                    "llm.prompt_tokens":
                        prompt_tokens,

                    "llm.completion_tokens":
                        completion_tokens,

                    "llm.total_tokens":
                        total_tokens,
                }
            )

        return answer