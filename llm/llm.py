from dataclasses import dataclass


class LLM:
    @dataclass
    class SimpleResponse:
        answer: str
        input_tokens: int
        output_tokens: int

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> SimpleResponse:
        raise NotImplementedError()

    def parallelism(self) -> int:
        raise NotImplementedError()

    def metadata(self) -> dict[str, object]:
        """Return non-secret provider details that make benchmark runs reproducible."""
        return {"provider": type(self).__name__}

    async def close(self) -> None:
        """Release provider resources; stateless test providers need no cleanup."""
