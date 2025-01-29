import itertools
import os
import typing as t
from pathlib import Path

import jsonlines
import polars
import srsly
import typer
from datasets import Dataset, concatenate_datasets, load_dataset

from . import schema
from .utils import autotrain_project_name_for_model_id


def _parse_train_args(train_args: list[str]) -> dict[str, t.Any]:
    args = []
    for idx, arg in enumerate(train_args[:]):
        if (idx + 1) < len(train_args):
            next_arg = train_args[idx + 1]
        else:
            next_arg = ""
        if arg.startswith("--"):
            args.append(arg[2:])
            if next_arg.startswith("--"):
                args.append("true")
        else:
            args.append(arg)
    return dict(item for item in itertools.pairwise(args) if item[0].startswith("--"))


def write(data_file: Path, items: t.Iterable[dict[str, str]] | Dataset) -> None:
    data_file.parent.mkdir(exist_ok=True, parents=True)
    with open(data_file, "wt") as fp:
        with jsonlines.Writer(fp) as writer:
            writer.write_all(items)


def generate_items(
    dataset: t.Any, text_column: str, split: str | None = None
) -> t.Iterator[dict[str, str]]:
    items = dataset[split] if split is not None else dataset
    for entry in items:
        yield {"text": entry[text_column]}


def configure_autotrain(
    dataset_dir: Path,
    base_model: str,
    task: str,
    push_to_hub: bool = False,
    **train_params,
) -> schema.AutoTrainConfig:
    hf_username = os.environ.get("HF_USER")
    hf_token = os.environ.get("HF_TOKEN")
    hub = schema.Hub(username=hf_username, token=hf_token, push_to_hub=push_to_hub)
    params = schema.TrainParams(**train_params)  # type: ignore
    project_name = autotrain_project_name_for_model_id(base_model)
    cfg = schema.AutoTrainConfig(
        base_model=base_model,
        project_name=project_name,
        task=task,
        data=schema.DataConfig(path=str(dataset_dir)),
        hub=hub,
        params=params,
    )
    return cfg


def build_ctp_dataset(datasources_config: Path) -> Dataset:
    data_sources = schema.DataSources(**srsly.read_yaml(datasources_config))  # type: ignore
    datasets = []
    for dataset_spec in data_sources.datasets:
        ds = load_dataset(dataset_spec.path, name=dataset_spec.config_name)
        ds = ds[dataset_spec.split].rename_column(dataset_spec.text_column, "text")  # type: ignore
        ds = ds.remove_columns(set(ds.column_names) - {"text"})  # type: ignore
        datasets.append(ds)
    for dataset_spec in data_sources.parallel_sentences:
        df = polars.read_csv(dataset_spec.path, separator="\t")
        df = df.rename({dataset_spec.text_column: "text"})
        df = df.drop(*tuple(set(df.columns) - {"text"}))
        ds = Dataset.from_polars(df)
        datasets.append(ds)
    # cast all datasets to have homogenous types (features) so they can be concatenated.
    # d_type: string vs d_type: large_string.
    features = datasets[-1].features
    homogenous_datasets = [ds.cast(features) for ds in datasets[:-1]]
    homogenous_datasets.append(datasets[-1])
    return concatenate_datasets(homogenous_datasets)


app = typer.Typer()
ctp = typer.Typer()
app.add_typer(ctp, name="ctp")


@ctp.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def configure(
    ctx: typer.Context,
    config: typer.FileTextWrite = typer.Option(...),
    base_model: str = typer.Option(default="meta-llama/Meta-Llama-3-8B-Instruct"),
    dataset_dir: Path = typer.Option(default="data/dataset"),
    task: str = typer.Option(default="llm"),
    push_to_hub: bool = typer.Option(default=False),
) -> None:
    train_params = _parse_train_args(ctx.args)
    at_config = configure_autotrain(
        dataset_dir, base_model, task, push_to_hub=push_to_hub, **train_params
    )
    config.write(srsly.yaml_dumps(at_config.model_dump()))


@ctp.command()
def build(
    ctx: typer.Context,
    datasources_config: Path = typer.Option(default="data-sources.yaml"),
    dataset_dir: Path = typer.Option(default="data/dataset"),
    dataset_filename: str = typer.Option(default="train.jsonl"),
) -> None:
    dataset_dir.mkdir(exist_ok=True)
    dataset = build_ctp_dataset(datasources_config)
    write(dataset_dir / dataset_filename, dataset)
