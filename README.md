# MIMiC - Modeulau Iaith Mawr i'r Cymraeg (Large Language Models for Welsh)

This package provides a utilities for performing training and inference with LLMs,
and a demo [gradio](https://www.gradio.app/) application showcasing the inference on trained models.

Each of the components is exposed as library functions but also have command line interface (CLI) counterpart.

For CLI usage, see the inline help provided by the CLI:

```bash
python -m techiaith.mimic --help
```

`techiaith-mimic` is developed and maintained by the Uned Technolegau Iaith (UTI) <https://techiaith.cymru/> team, backed by Bangor University.
<https://bangor.ac.uk/>.

Techiaith UTI is a self-funded research unit that develops technologies for the Welsh language.

To learn more about who specifically contributed to this codebase, see our contributors page.

This code is made available under the MIT License.

## Install

Install this package from PyPI with pip using a venv or mamba/conda environment:

```bash
pip install techiaith-mimic
```

or directly from [github](https://github.com/techiaith/mimic):

```bash
pip install git@github.com:techiaith/mimic.git
```

## Development

Configure your development environment to use the tools refered to in the `pyproject.toml` file (mypy, ruff, black).

### Releases - Publishing to PyPI

Change the version as appropriate in `pyproject.toml`, then build, check and upload:

```bash
pip install build twine 
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
