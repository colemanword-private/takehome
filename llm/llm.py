from dataclasses import dataclass


class LLMResponseError(RuntimeError):
    """Raised when a provider returns a response without usable answer text.

    Carries the billable token usage of the failed attempt so callers can
    observe cost from failures through one declared type instead of
    duck-typed attribute names.
    """

    def __init__(
        self, message: str, *, input_tokens: int, output_tokens: int
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


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
