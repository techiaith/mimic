import typing as t
from pathlib import Path

import pydantic


class FormatOperation(pydantic.BaseModel):
    out_dir: Path
    training_config: Path


class Message(pydantic.BaseModel):
    role: t.Literal["system"] | t.Literal["user"]
    content: str


class Conversation(pydantic.BaseModel):
    messages: list[Message]


class Segment(pydantic.BaseModel):
    text: str


class EvalInstruction(pydantic.BaseModel):
    instruction: str


class ColumnMapping(pydantic.BaseModel):
    text_column: str = "text"
    prompt_text_column: str | None = None  # e.g "question"


class DataConfig(pydantic.BaseModel):
    path: str
    chat_template: str | None = None
    train_split: str | None = "train"
    valid_split: str | None = None
    column_mapping: ColumnMapping = ColumnMapping()


class Hub(pydantic.BaseModel):
    username: str | None = None
    token: str | None = None
    push_to_hub: bool = False


class TrainParams(pydantic.BaseModel):
    block_size: int = 2048
    model_max_length: int = 8192
    epochs: int = 2
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
    hub: Hub
    params: TrainParams
    log: str = "dvclive"
    backend: str = "local"
    model_config = pydantic.ConfigDict(extra="allow")


class DatasetSource(pydantic.BaseModel):
    path: str
    text_column: str = "text"
    split: str = "train"
    config_name: str = "default"


class TSVSource(pydantic.BaseModel):
    path: str
    text_column: str = "text"


class DataSources(pydantic.BaseModel):
    datasets: list[DatasetSource] = []
    parallel_sentences: list[TSVSource] = []
