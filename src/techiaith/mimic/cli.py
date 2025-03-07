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


if __name__ == "__main__":
    app()
