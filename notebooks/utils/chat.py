from anthropic import Anthropic, omit, Omit
from anthropic._types import SequenceNotStr
from anthropic.types import MessageParam

client = Anthropic()


def add_user_message(messages: list[MessageParam], text: str) -> None:
    messages.append(MessageParam(role="user", content=text))


def add_assistant_message(messages: list[MessageParam], text: str) -> None:
    messages.append(MessageParam(role="assistant", content=text))


def chat(
    messages: list[MessageParam],
    model: str = "claude-sonnet-5",
    max_tokens: int = 1000,
    system: str | Omit = omit,
    stop_sequences: SequenceNotStr[str] | Omit = omit,
    output_config: dict | Omit = omit,
) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        stop_sequences=stop_sequences,
        output_config=output_config,
    )
    return next(block.text for block in message.content if block.type == "text")
