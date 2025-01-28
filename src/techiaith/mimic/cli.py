import typer

from . import dataset, inference

app = typer.Typer()
app.add_typer(dataset.app, name="dataset")
app.add_typer(inference.app, name="inference")

if __name__ == "__main__":
    app()
