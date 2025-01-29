import srsly

def autotrain_project_name_for_model_id(base_model: str, prefix="") -> str:
    name = "".join(ch if any([ch.isalnum(), ch == "-"]) else "--" for ch in base_model)
    return f"{prefix}{name}"

def techiaith_model_name(suffix: str = "ctp-cy") -> str:
    params = srsly.read_yaml("params.yaml")
    model_name = params["base_model"].split("/")[-1].lower()
    return f"techiaith/{model_name}-{suffix}"
