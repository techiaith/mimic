import collections
import json
import typing as t
from pathlib import Path

import jsonlines
import polars
import srsly
import torch
import transformers
import typer

from .schema import AutoTrainConfig

START_INSTRUCTION: str = "[INST]"
END_INSTRUCTION: str = "[/INST]"
SENT_START: str = "<s>"
SENT_END: str = "</s>"


def _get_tokenizer(
    model_id: str,
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
    return tokenizer


def _make_chatbot_conversation(
    user_message: str, system_message: str = ""
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return messages


def chatbot(
    model_id: str, messages: list[dict[str, str]], max_new_tokens: int = 512
) -> t.Any:
    tokenizer = _get_tokenizer(model_id)
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe(messages, max_new_tokens=max_new_tokens)


@t.no_type_check
def format_chatbot_reply(replies: list[str]) -> str | None:
    if not replies:
        return None
    content = None
    for message in replies[0]["generated_text"]:
        role = message["role"]
        if role == "assistant":
            content = message["content"]
            break
    return content


app = typer.Typer()


@app.command(name="chatbot")
def _chatbot(model_id: str, conversation: str) -> None:
    messages = json.loads(conversation)
    print(chatbot(model_id, messages))


@app.command()
def eval_instructions(
    ctx: typer.Context,
    trained_model: Path,
    eval_instructions_cy_file: Path,
    eval_instructions_en_file: Path,
    autotrain_config_file: Path = typer.Option(default="autotrain.yaml"),
    out_dir: Path = typer.Option(default="evals/output"),
    system_message: t.Annotated[str, typer.Option()] = "",
) -> None:
    autotrain_config = AutoTrainConfig(**srsly.read_yaml(autotrain_config_file))  # type: ignore
    out_dir.mkdir(exist_ok=True)
    out_filename = out_dir / f"{autotrain_config.project_name}.csv"
    out_file = out_dir / out_filename
    data = collections.defaultdict(list)
    questions = []
    for eval_input_file in (eval_instructions_cy_file, eval_instructions_en_file):
        with jsonlines.open(eval_input_file) as jl:
            questions.extend(jl)
    models = list(map(str, (autotrain_config.base_model, trained_model)))
    for model, model_label in zip(models, ("base_model", "trained_model")):
        for message in questions:
            data["model"].append(model_label)
            data["question"].append(message["content"])
            messages = _make_chatbot_conversation(message["content"], system_message)
            replies = chatbot(model, messages=messages)
            answer = format_chatbot_reply(replies)
            data["answer"].append(answer)
    df = polars.DataFrame(data)
    df.write_csv(out_file)
    print(df)


if __name__ == "__main__":
    app()
