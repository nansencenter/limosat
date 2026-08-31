import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_xfeat_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    for name in ("buoy_descriptor_benchmark", "orb_multiframe_graph"):
        path = experiments / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    path = experiments / "xfeat_buoy_graph.py"
    spec = importlib.util.spec_from_file_location("xfeat_buoy_graph", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resize_for_xfeat_preserves_coordinate_scale_and_multiple_of_32():
    module = load_xfeat_module()
    image = np.zeros((1000, 2000), dtype=np.uint8)

    resized, sx, sy = module.resize_for_xfeat(image, max_side=1024)

    assert resized.shape == (512, 1024)
    assert sx == 2000 / 1024
    assert sy == 1000 / 512
