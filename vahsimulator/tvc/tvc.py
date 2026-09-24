from .tvc_base import TVCBase
from .tvc_simplified import TVCSimplified
from .tvc_detailed import TVCDetailed


TVC_MODELS = {
    "tvc_simplified": TVCSimplified,
    "tvc_detailed": TVCDetailed,
}


def tvc_factory(config: dict, dt: float) -> TVCBase:

    tvc_config = config["tvc_parameters"]
    phase_id = 0
    model_name = tvc_config[phase_id]["type"]

    try:
        tvc_class = TVC_MODELS[model_name]

    except KeyError:
        raise ValueError(
            f"Unknown TVC model: {model_name!r}. "
            f"Available models: {list(TVC_MODELS)}"
        )

    return tvc_class(dt=dt)
