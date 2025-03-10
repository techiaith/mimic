import typing as t
from pathlib import Path

import transformers
import typer
from langchain import chat_models


class Provider:
    def __init__(
        self,
        model_name: str,
        model_provider: str,
        user_prompt_template: str = "",
        system_message: str = "",
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.user_prompt_template = user_prompt_template
        self.system_message = system_message
        if self.model_provider == "transformers":
            model = transformers.pipeline("text-generation", model=self.model_name)
        else:
            model = chat_models.init_chat_model(
                self.model_name, model_provider=self.model_provider
            ).invoke
        self.model = model

    def __call__(self, text: str) -> str:
        messages = self.make_messages(
            text,
            user_prompt_template=self.user_prompt_template,
            system_message=self.system_message,
        )
        return self.invoke(messages)

    @classmethod
    def handle_transformers_response(cls, response: list[dict]) -> str | None:
        if response:
            for message in response[0]["generated_text"]:
                role = message["role"]
                if role == "assistant":
                    content = message["content"]
                    return content
        return None

    @classmethod
    def handle_anthropic_response(cls, response) -> str | None:
        return getattr(response, "content", None)

    @classmethod
    def make_messages(
        self, text: str, user_prompt_template: str = "", system_message: str = ""
    ) -> list[dict[str, str]]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if user_prompt_template:
            text = user_prompt_template.format(text=text)
        messages.append({"role": "user", "content": text})
        return messages

    def invoke(self, messages: list[dict[str, str]]) -> str | None:
        response = self.model(messages)
        format_response = getattr(self, f"handle_{self.model_provider}_response")
        return format_response(response)


app = typer.Typer()


@app.command(name="chatbot")
def _chatbot(
    model_id: str,
    model_provider: t.Annotated[
        str, typer.Option("-p", '--model-provider"')
    ] = "transformers",
    user_prompt_template_file: t.Annotated[
        Path, typer.Option("-t", "--user-prompt-template-file")
    ] = None,
    system_message: str = "",
) -> None:
    text = typer.prompt("Enter the text to send to the chatbot:")
    if user_prompt_template_file:
        user_prompt_template = user_prompt_template_file.read_text()
    messages = Provider.make_messages(
        text, user_prompt_template=user_prompt_template, system_message=system_message
    )
    chatbot = Provider(model_id, model_provider)
    print(chatbot.invoke(messages))


if __name__ == "__main__":
    app()
