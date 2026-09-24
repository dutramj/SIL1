# Python standard libraries
from pathlib import Path

# 3rd party libraries
import numpy as np
from numpy.typing import DTypeLike
import pandas as pd
from typing_extensions import Any
import yaml

# VAHSimulator library
from .optimize import OptimizationVariable

def _yaml_constructor(loader: yaml.Loader, node: yaml.ScalarNode) -> Any:
    path_str = loader.construct_scalar(node)
    base_path = Path(loader.name).parent
    path = (base_path / path_str).resolve()

    with open(path) as f:
        sub_loader = yaml.SafeLoader(f)
        sub_loader.name = str(path)
        return sub_loader.get_single_data()
    

def _csv_constructor(loader: yaml.Loader, node: yaml.ScalarNode) -> pd.DataFrame:
    path_str: str = loader.construct_scalar(node)

    base_path = Path(loader.name).parent
    path = (base_path / path_str).resolve()

    return pd.read_csv(path)


def _eq_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
    equation = loader.construct_scalar(node)
    return eval(equation,{"np": np})


def _opt_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
    opt_lims = loader.construct_sequence(node)
    return OptimizationVariable.from_list(opt_lims)


def _convert_floats(obj: Any, dtype: DTypeLike = np.float64) -> Any:
    if isinstance(obj, float):
        return dtype(obj)

    if isinstance(obj, dict):
        return {k: _convert_floats(v, dtype) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_convert_floats(v, dtype) for v in obj]

    return obj


def load_config(mission_file: Path) -> dict:

    # Constructors
    yaml.SafeLoader.add_constructor("!yaml", _yaml_constructor)
    yaml.SafeLoader.add_constructor("!csv", _csv_constructor)
    yaml.SafeLoader.add_constructor("!eq", _eq_constructor)
    yaml.SafeLoader.add_constructor("!opt", _opt_constructor)

    # Load YAML file
    mission_config = yaml.safe_load(open(mission_file))

    # Conversion into np.float64
    mission_config = _convert_floats(mission_config, np.float64)

    return mission_config
