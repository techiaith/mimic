import typing as t
from collections import defaultdict
from pathlib import Path

import jsonlines
import langcodes
import polars as pl
import srsly
import typer
from datasets import Dataset, concatenate_datasets, load_dataset

from . import dataset_metadata
from .schema import DataSources


def write(data_file: Path, items: t.Iterable[dict[str, str]] | Dataset) -> None:
    data_file.parent.mkdir(exist_ok=True, parents=True)
    with open(data_file, "wt") as fp:
        with jsonlines.Writer(fp) as writer:
            writer.write_all(items)


def concatenate(datasets: list[Dataset]) -> Dataset:
    """Concatenate datasets.

    Casts all datasets to have homogenous types (features) so they can be concatenated.
    """
    features = datasets[-1].features
    homogenous_datasets = [dataset.cast(features) for dataset in datasets[:-1]]
    homogenous_datasets.append(datasets[-1])
    return concatenate_datasets(homogenous_datasets)


def build_monolingual_dataset(data_sources: DataSources) -> Dataset:
    datasets = []
    for dataset_spec in data_sources.datasets:
        dataset_dict = load_dataset(dataset_spec.path, name=dataset_spec.config_name)
        dataset = (
            dataset_dict[dataset_spec.split]
            .rename_column(dataset_spec.text_column, "text")
            .select_columns(["text"])
        )
        datasets.append(dataset)
    for datasource_spec in data_sources.parallel_sentences:
        df = pl.read_csv(datasource_spec.path, separator="\t")
        dataset = (
            Dataset.from_polars(df)
            .rename_column(datasource_spec.text_column, "text")
            .select_columns(["text"])
        )
        datasets.append(dataset)
    return concatenate(datasets)


def build_bilingual_dataset(
    data_sources: DataSources,
    src_lang: langcodes.Language,
    trg_lang: langcodes.Language,
) -> Dataset:
    datasets = []
    text_template = "{}: {}\n{}: {}"
    data = defaultdict(list)
    src_lang_name, trg_lang_name = src_lang.language_name(), trg_lang.language_name()
    for dataset_spec in data_sources.datasets:
        dataset_dict = load_dataset(dataset_spec.path, name=dataset_spec.config_name)
        dataset = dataset_dict[dataset_spec.split]
        dataset = dataset.select_columns(
            [dataset_spec.text_column, dataset_spec.target_column]
        )
        texts = dataset[dataset_spec.text_column]
        translations = dataset[dataset_spec.target_column]
        for text, translation in zip(texts, translations, strict=True):
            text = text_template.format(src_lang_name, text, trg_lang_name, translation)
            data["text"].append(text)
        dataset = Dataset.from_dict(data)
        datasets.append(dataset)
    return concatenate(datasets)


def build_dataset(
    datasources_config: Path, src_lang: str = "", trg_lang: str = ""
) -> Dataset:
    langs = {
        lang: langcodes.Language.get(lang) for lang in (src_lang, trg_lang) if lang
    }
    if langs:
        return build_bilingual_dataset(
            datasources_config, langs[src_lang], langs[trg_lang]
        )
    return build_monolingual_dataset(datasources_config)


app = typer.Typer()
app.add_typer(dataset_metadata.app, name="metadata")


@app.command()
def build(
    datasources_config: Path = typer.Option(default="data-sources.yaml"),
    dataset_dir: Path = typer.Option(default="data/dataset"),
    dataset_filename: str = typer.Option(default="train.jsonl"),
    src_lang: str = "",
    trg_lang: str = "",
) -> None:
    dataset_dir.mkdir(exist_ok=True)
    data_sources = DataSources(**srsly.read_yaml(datasources_config))
    dataset = build_dataset(data_sources, src_lang=src_lang, trg_lang=trg_lang)
    write(dataset_dir / dataset_filename, dataset)
