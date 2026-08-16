import os

from together import AsyncTogether
from together.types.chat.completion_create_params import (
    MessageChatCompletionSystemMessageParam,
    MessageChatCompletionUserMessageParam,
)

from .llm import LLM


class Together(LLM):
    def __init__(
        self,
        model: str | None = None,
        *,
        client: AsyncTogether | None = None,
    ) -> None:
        self.__model = model or os.getenv("TOGETHER_MODEL")
        if not self.__model:
            raise ValueError("TOGETHER_MODEL must be set")
        api_key = os.getenv("TOGETHER_API_KEY")
        if client is None and not api_key:
            raise ValueError("TOGETHER_API_KEY must be set")
        self.__client = client or AsyncTogether(api_key=api_key)

    def parallelism(self) -> int:
        return 100

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "Together",
            "model": self.__model,
            "parallelism": self.parallelism(),
        }

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        response = await self.__client.chat.completions.create(
            model=self.__model,
            messages=[
                MessageChatCompletionSystemMessageParam(role="system", content=system_prompt),
                MessageChatCompletionUserMessageParam(role="user", content=question),
            ],
            logprobs=1,
            temperature=temperature,
        )

        return LLM.SimpleResponse(
            answer=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    async def close(self) -> None:
        await self.__client.close()
