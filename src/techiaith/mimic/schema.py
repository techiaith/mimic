import typing as t
from pathlib import Path

import pydantic


class ColumnMapping(pydantic.BaseModel):
    text_column: str = "text"
    prompt_text_column: str | None = None  # e.g "question"


class DataConfig(pydantic.BaseModel):
    path: str
    chat_template: str | None = None
    train_split: str | None = "train"
    valid_split: str | None = None
    column_mapping: ColumnMapping = ColumnMapping()


class HubConfig(pydantic.BaseModel):
    username: str | None = None
    token: str | None = None
    push_to_hub: bool = False


class TrainingParams(pydantic.BaseModel):
    block_size: int = 2048
    model_max_length: int = 8192
    epochs: int = 4
    batch_size: int = 1
    lr: float = 1e-5
    peft: bool = True
    quantization: str | None = None
    target_modules: str = "all-linear"
    padding: t.Literal["left"] | t.Literal["right"] | None = "right"
    optimizer: str | None = "paged_adamw_8bit"
    scheduler: str | None = "linear"
    gradient_accumulation: int = 8
    mixed_precision: str = "bf16"
    model_config = pydantic.ConfigDict(extra="allow", protected_namespaces=())


class AutoTrainConfig(pydantic.BaseModel):
    base_model: str
    project_name: str
    task: str
    data: DataConfig
    hub: HubConfig
    params: TrainingParams
    log: str = "dvclive"
    backend: str = "local"
    model_config = pydantic.ConfigDict(extra="allow")


class DataSource(pydantic.BaseModel):
    path: str
    text_column: str = "text"
    target_column: str | None = None


class DatasetSource(DataSource):
    config_name: str = "default"
    split: str = "train"


class DataSources(pydantic.BaseModel):
    datasets: list[DatasetSource] = []
    parallel_sentences: list[DataSource] = []
