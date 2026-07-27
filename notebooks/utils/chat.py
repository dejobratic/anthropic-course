from anthropic import Anthropic, omit, Omit
from anthropic._types import SequenceNotStr
from anthropic.types import MessageParam, OutputConfigParam

client = Anthropic()


class Chat:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        system: str | Omit = omit,
        max_tokens: int = 1000,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.messages: list[MessageParam] = []

    def user(self, text: str) -> "Chat":
        self.messages.append(MessageParam(role="user", content=text))
        return self

    def assistant(self, text: str) -> "Chat":
        self.messages.append(MessageParam(role="assistant", content=text))
        return self

    def send(
        self,
        *,
        prefill: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: SequenceNotStr[str] | Omit = omit,
        output_config: OutputConfigParam | Omit = omit,
    ) -> str:
        if prefill is not None:
            self.assistant(prefill)

        message = client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=self.messages,
            system=self.system,
            temperature=self.temperature if temperature is None else temperature,
            stop_sequences=stop_sequences,
            output_config=output_config,
        )
        text = next(block.text for block in message.content if block.type == "text")

        # Merge prefill + completion into a single valid assistant turn.
        self.messages[len(self.messages) - (1 if prefill is not None else 0):] = [
            MessageParam(role="assistant", content=(prefill or "") + text)
        ]
        return text
