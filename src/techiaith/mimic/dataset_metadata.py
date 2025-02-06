import typing as t
from collections import Counter, defaultdict

import datasets
import polars as pl
import typer
from huggingface_hub import hf_api

DatasetMetadata = dict[str, t.Any] | dict[str, str]


def parse_dataset_ids(ds_ids_file: t.TextIO) -> list[str]:
    lines = list(filter(None, map(str.strip, ds_ids_file.readlines())))
    ds_ids = [
        url_or_id.replace("https://huggingface.co/datasets/", "") for url_or_id in lines
    ]
    valid_ds_ids = [ds_id for ds_id in ds_ids if len(ds_id.split("/")) == 2]
    invalid = ",".join(sorted(set(ds_ids) - set(valid_ds_ids)))
    assert not invalid, f"Invalid dataset ids in file: {invalid}"
    return ds_ids


def get_collection_note(project_name: str, dataset_id: str) -> str:
    ds_collections = (
        coll
        for coll in hf_api.list_collections(
            owner=project_name, item=f"datasets/{dataset_id}"
        )
    )
    coll = next(ds_collections, None)
    if coll is not None:
        collection = hf_api.get_collection(coll.slug)
        note = next(
            (item.note for item in collection.items if item.item_id == dataset_id)
        )
        return note
    return "N/A"


def metadata_from_card_data(card_data: hf_api.DatasetCardData) -> DatasetMetadata:
    languages = ",".join(card_data["language"]) if card_data.get("language") else ""
    task_ids = ",".join(card_data["task_ids"]) if card_data.get("task_ids") else ""
    task_categories = (
        ",".join(card_data["task_categories"])
        if card_data.get("task_categories")
        else ""
    )
    licenses = (
        ",".join(card_data["license"])
        if isinstance(card_data["license"], list)
        else card_data["license"]
    )
    return dict(
        languages=languages,
        licenses=licenses,
        task_ids=task_ids,
        task_categories=task_categories,
    )


def metadata_from_project_dataset(ds_id: str) -> DatasetMetadata:
    ds_info = hf_api.dataset_info(ds_id)
    card_data = ds_info.card_data.to_dict() if ds_info.card_data else {}
    dataset_size = card_data["dataset_info"]["dataset_size"]
    metadata_item = dict(
        dataset=ds_id,
        dataset_size=dataset_size,
    )
    if ds_info.card_data is not None:
        metadata_item.update(metadata_from_card_data(ds_info.card_data).items())
    return metadata_item


def metadata_from_dataset(ds_id: str) -> DatasetMetadata:
    dataset_dict = datasets.load_dataset(ds_id)
    first_split = next(iter(dataset_dict))
    ds_info = hf_api.dataset_info(ds_id)
    metadata_item = dict(
        dataset=ds_id,
        dataset_size=dataset_dict[first_split].info.dataset_size,  # type: ignore
    )
    if ds_info.card_data is not None:
        metadata_item.update(metadata_from_card_data(ds_info.card_data))
    return metadata_item


app = typer.Typer()


@app.command()
def table(
    dataset_ids_file: t.Annotated[
        typer.FileText,
        typer.Option(
            "--dataset-ids-file",
            "-i",
            help="File containing HuggingFace dataset URLs or ids",
        ),
    ],
    write_tsv: bool = True,
):
    ds_info = defaultdict(list)
    dataset_ids = parse_dataset_ids(dataset_ids_file)
    owners = Counter(ds_id.split("/")[0] for ds_id in dataset_ids)
    primary_project_name = owners.most_common(1)[0][0]
    for ds_id in dataset_ids:
        owner = ds_id.split("/")[0]
        get_metadata = (
            metadata_from_project_dataset
            if owner == primary_project_name
            else metadata_from_dataset
        )
        metadata = get_metadata(ds_id)
        metadata["note"] = get_collection_note(primary_project_name, ds_id)
        for k, v in metadata.items():
            ds_info[k].append(v)
    outfile_name = f"{primary_project_name}_dataset_metadata.tsv"
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        df = pl.DataFrame(ds_info)
        df = df.sort(by="dataset_size", descending=False)
        print(df)
        if write_tsv:            
            df.write_csv(outfile_name, include_header=True, separator="\t")
            print("Output written to file:", outfile_name)
