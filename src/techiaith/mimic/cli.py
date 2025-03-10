from pathlib import Path

import srsly
import typer

from . import dataset, evals, inference
from .utils import autotrain_project_name, configure_autotrain, parse_training_args

app = typer.Typer()
app.add_typer(dataset.app, name="dataset")
app.add_typer(evals.app, name="evals")
app.add_typer(inference.app, name="inference")

train = typer.Typer()
app.add_typer(train, name="train")


@train.command()
def project_name(base_model_id: str) -> None:
    print(autotrain_project_name(base_model_id))


@train.command()
def setup(base_model_id: str) -> None:
    project_name = autotrain_project_name(base_model_id)
    params = {"base_model": base_model_id, "project_name": project_name}
    srsly.write_yaml("params.yaml", params)


@train.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def configure(
    ctx: typer.Context,
    project_name: str,
    autotrain_task: str = typer.Option(default="llm-sft"),
    config: typer.FileTextWrite = typer.Option(...),
    base_model: str = typer.Option(default="meta-llama/Meta-Llama-3-8B-Instruct"),
    dataset_dir: Path = typer.Option(default="data/dataset"),
    push_to_hub: bool = typer.Option(default=False),
) -> None:
    training_params = parse_training_args(ctx.args)
    configure_autotrain(
        project_name,
        config,
        dataset_dir,
        base_model,
        autotrain_task,
        push_to_hub=push_to_hub,
        **training_params,
    )


if __name__ == "__main__":
    app()
