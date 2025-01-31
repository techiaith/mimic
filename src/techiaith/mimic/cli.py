from pathlib import Path
import huggingface_hub
import srsly
import typer

from . import dataset, inference, utils

app = typer.Typer()
app.add_typer(dataset.app, name="dataset")
app.add_typer(inference.app, name="inference")

train = typer.Typer()
app.add_typer(train, name="train")

@train.command()
def project_name(base_model_id: str) -> None:
    print(utils.autotrain_project_name_for_model_id(base_model_id))


@train.command()
def setup(base_model_id: str) -> None:
    params = srsly.read_yaml("params.yaml")
    params["base_model"] = base_model_id
    params["project_name"] = utils.autotrain_project_name_for_model_id(base_model_id)
    srsly.write_yaml("params.yaml", params)


@train.command()
def upload_model_to_hf(name_suffix: str = "ctp-cy") -> None:
    repo_id = utils.techiaith_model_name()
    model_path = utils.trained_model_path()
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if not token_path.is_file():
        raise RuntimeError("HuggingFace token not found, please run `huggingface-cli login`")
    token = token_path.read_text()
    huggingface_hub.upload_folder(
        repo_id=repo_id, folder_path=model_path, repo_type="model", token=token
    )


if __name__ == "__main__":
    app()
