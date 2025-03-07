import typing as t
from pathlib import Path

import jsonlines
import gradio as gr
import typer

from .inference import chatbot

example_system_prompts: list[list[str]] = []
for language in ("Welsh", "English"):
    example_system_prompts.append([f"You must answer in {language}."])
    example_system_prompts.append(
        [f"You must answer in {language}, be brief and do not elaborate"]
    )

class Defaults(t.NamedTuple):
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    system_prompt: str = example_system_prompts[0][0]


def ask_question(model_id_or_path: str, question: str, system_prompt_text: str) -> str:
    questions = [
        dict(role="system", content=system_prompt_text),
        dict(role="user", content=question),
    ]
    response = chatbot(model_id_or_path, questions)
    answers = []
    for entry in response[0]["generated_text"]:
        if entry["role"] == "assistant":
            answers.append(dict(text=entry["content"]))
    return answers[0]["text"]


def q_and_a(
    question: str,
    model_id_or_path: str,
    system_prompt_text: str,
) -> str:
    answer = ask_question(model_id_or_path, question, system_prompt_text)
    return f"<p><dl><dt>Ateb:</dt><dd>{answer}</dd></dl></p>"


def format_example_questions(example_questions_files: list[Path]) -> list:
    samples = []
    for example_questions_file in example_questions_files:
        with jsonlines.open(example_questions_file) as jl:
            samples.extend([[message["content"]] for message in jl])
    return samples


defaults = Defaults()

HTML_OUTPUT: str = "<div style='height: 200px; width: 200px;'></div>"

TITLE: str = "MIMiC: Modelau Iaith Mawr i'r Cymraeg"


hf_model_url: t.Callable = "https://huggingface.co/{model_id}".format


def main(
    example_questions_file: t.Annotated[
        t.Optional[list[Path]], typer.Option("--example-questions-file", "-e")
    ] = None,
    model_id_or_path: str = defaults.model_id,
    system_prompt_text: str = defaults.system_prompt,
):
    with gr.Blocks(theme=gr.themes.Ocean(), title=TITLE) as dmeo:
        model_id = gr.Textbox(value=model_id_or_path, visible=False)
        with gr.Row():
            gr.HTML(f"<h2>{TITLE}</h2>")
        with gr.Row():
            model = model_id_or_path
            local_model = Path(model_id_or_path)
            if local_model.is_dir():
                where = "local"
                model = local_model.name
            else:
                where = ""
                model_url = hf_model_url(model_id=model_id_or_path)
                model = f'<a href="{model_url}">{model_id_or_path}</a>'
            where = f"{where} HuggingFace"
            gr.HTML(
                f"Demo yn arddangos defnydd o'r fodel <b>{model}</b> {where} i'r dasg cwestiwn ac ateb."
            )
        with gr.Row():
            question = gr.Textbox(
                label="Cwestiwn",
                placeholder="Teipiwch eich cwestiwn yn Gymraeg neu Saesneg yma...",
            )
            if example_questions_file is not None and all(
                f.is_file() for f in example_questions_file
            ):
                example_questions = format_example_questions(example_questions_file)
            else:
                example_questions = []
        if example_questions:
            with gr.Row():
                gr.Examples(
                    label="Cwestiynau enghreifftiol",
                    examples=example_questions,
                    inputs=[question],
                )
            with gr.Row():
                gr.HTML("<hr>")
        with gr.Row():
            system_prompt = gr.Textbox(
                label="Neges system",
                placeholder=example_system_prompts[0][0],
                value=example_system_prompts[0][0],
            )
            with gr.Row():
                gr.Examples(
                    example_system_prompts,
                    inputs=[system_prompt],
                    label="Negesau system enghreifftiol",
                )
        inputs = [question, model_id, system_prompt]
        with gr.Row(variant="compact"):
            ask_btn = gr.Button(
                value="Gofyn", inputs=inputs, size="sm", variant="secondary"
            )
        with gr.Row():
            output = gr.HTML(
                value=f"<hr>{HTML_OUTPUT}",
                label="Ateb",
            )

        @ask_btn.click(inputs=inputs, outputs=[output])
        @question.submit(inputs=inputs, outputs=[output])
        def on_ask(*args) -> str:
            answer = q_and_a(*args)
            return answer

    dmeo.launch()


if __name__ == "__main__":
    typer.run(main)
