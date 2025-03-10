import collections
import datetime
import typing as t
from argparse import Namespace
from pathlib import Path

import polars as pl
import jsonlines
import sacrebleu
import srsly
import typer
from datachain import DataChain
from datasets import Dataset

from .schema import AutoTrainConfig, ModelProvider
from .inference import Provider as InferenceProvider

app = typer.Typer()
translation = typer.Typer()
app.add_typer(translation, name="translation")


@app.command()
def q_and_a(
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
    chatbot = InferenceProvider(autotrain_config.base_model, "transformers")
    questions = []
    for eval_input_file in (eval_instructions_cy_file, eval_instructions_en_file):
        with jsonlines.open(eval_input_file) as jl:
            questions.extend(jl)
    models = list(map(str, (autotrain_config.base_model, trained_model)))
    for model, model_label in zip(models, ("base_model", "trained_model")):
        for message in questions:
            data["model"].append(model_label)
            data["question"].append(message["content"])
            messages = InferenceProvider.make_messages(
                message["content"], system_messag=system_message
            )
            answer = chatbot(messages)
            data["answer"].append(answer)
    df = pl.DataFrame(data)
    df.write_csv(out_file)
    print(df)


def tsv_column_values(tsv_file: t.TextIO | Path, column: str) -> list[str]:
    return list(pl.read_csv(tsv_file, separator="\t").select([column]).to_series())


@translation.callback()
def translation_options(
    ctx: typer.Context,
    gold_testset_tsv: Path = typer.Option(
        help="The path to the gold test set TSV file."
    ),
    references_column: str = typer.Option(
        help="The name of the column in the gold test set that contains the references."
    ),
    sources_column: str = typer.Option(
        help="The name of the column in the gold test set that contains the sources to translate."
    ),
    model: str = typer.Option(
        help="The model to use for translation. Can be a HF model id or a langchain model name."
    ),
    model_provider: t.Annotated[
        ModelProvider, typer.Option()
    ] = ModelProvider.transformers,
    system_message: str = typer.Option(
        default="", help="The system message to use for translating with the LLM."
    ),
) -> None:
    gold_testset = Dataset.from_csv(str(gold_testset_tsv), delimiter="\t", split="test")
    gold_testset = gold_testset.select_columns([references_column, sources_column])
    ctx.obj = Namespace(
        model=model,
        model_provider=model_provider,
        system_message=system_message,
        gold_testset=gold_testset,
        sources_column=sources_column,
        references_column=references_column,
        references=tsv_column_values(gold_testset_tsv, references_column),
        sources=tsv_column_values(gold_testset_tsv, sources_column),
    )


@translation.command()
def register_model(
    model: str,
    model_provider: str,
    output_label: str,
    targets_file: Path = typer.Option("evals/targets.json"),
) -> None:
    eval_targets = srsly.json_loads(targets_file.read_text())
    target = {
        "model": model,
        "model_provider": model_provider,
        "output_label": output_label,
    }
    models = eval_targets["models"]
    if target not in models:
        models.append(target)
        srsly.write_json(eval_targets, targets_file)


@translation.command()
def generate_hypothoses(
    ctx: typer.Context,
    output: t.Annotated[typer.FileTextWrite, typer.Option("--output", "-o")],
    user_prompt_template_file: t.Annotated[
        Path, typer.Option("--user-prompt-template", "-p")
    ] = None,
    limit: int = typer.Option(default=4000),
) -> None:
    sources_column = ctx.obj.sources_column
    user_prompt_template = user_prompt_template_file.read_text()
    translator = InferenceProvider(
        ctx.obj.model,
        ctx.obj.model_provider,
        user_prompt_template=user_prompt_template,
        system_message=ctx.obj.system_message,
    )
    testset = ctx.obj.gold_testset.select_columns(ctx.obj.sources_column)
    chain = (
        DataChain.from_hf(testset)
        .limit(limit)
        .map(translation=translator, params=[sources_column])
        .select("translation")
    )
    chain.show()
    chain.to_csv(output.name, delimiter="\t")
    # df = pl.DataFrame({"translations": translations})
    # df.write_csv(output, include_header=True, separator="\t")


@translation.command()
def score(
    ctx: typer.Context,
    hypotheses_file: t.Annotated[typer.FileText, typer.Option()],
    results_file: t.Annotated[typer.FileTextWrite, typer.Option()],
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "model": ctx.obj.model,
        "model_provider": ctx.obj.model_provider,
        "system_message": ctx.obj.system_message,
        "timestamp": timestamp,
    }
    hypotheses = tsv_column_values(hypotheses_file, "translations")
    references = ctx.obj.references
    for metric in ("bleu", "ter", "chrf"):
        corpus_metric = getattr(sacrebleu, f"corpus_{metric}")
        metric_score = corpus_metric(hypotheses, references)
        results[metric] = f"{metric_score.score:0.2f}"
    with jsonlines.open(results_file, "a") as jl:
        jl.write(results)
