import itertools
import os
import typing as t
from pathlib import Path

import srsly

from .schema import AutoTrainConfig, DataConfig, HubConfig


def autotrain_project_name(base_model: str, prefix: str = "") -> str:
    name = "".join(ch if any([ch.isalnum(), ch == "-"]) else "--" for ch in base_model)
    return f"{prefix}{name}"


def parse_training_args(train_args: list[str]) -> dict[str, t.Any]:
    args = []
    for idx, arg in enumerate(train_args[:]):
        next_idx = idx + 1
        if next_idx < len(train_args):
            next_arg = train_args[next_idx]
        else:
            next_arg = ""
        if arg.startswith("--"):
            args.append(arg[2:])
            if next_arg.startswith("--"):
                args.append("true")
        else:
            args.append(arg)
    return dict(item for item in itertools.pairwise(args) if item[0].startswith("--"))


def configure_autotrain(
    project_name: str,
    config: Path,
    dataset_dir: Path,
    base_model: str,
    task: str,
    push_to_hub: bool = False,
    **train_params,
) -> None:
    hf_username = os.environ.get("HF_USER")
    hf_token = os.environ.get("HF_TOKEN")
    hub = HubConfig(username=hf_username, token=hf_token, push_to_hub=push_to_hub)
    at_config = AutoTrainConfig(
        base_model=base_model,
        project_name=project_name,
        task=task,
        data=DataConfig(path=str(dataset_dir)),
        hub=hub,
        params=train_params,
    )
    config.write(srsly.yaml_dumps(at_config.model_dump()))
