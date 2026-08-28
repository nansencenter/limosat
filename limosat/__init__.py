# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
LiMOSAT: A Python package for sea ice drift analysis.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .image import Image
    from .image_processor import ImageProcessor
    from .keypoint_detector import KeypointDetector
    from .keypoints import Keypoints
    from .matcher import Matcher
    from .templates import Templates

_PUBLIC_MODULES = {
    "ImageProcessor": ".image_processor",
    "Image": ".image",
    "Keypoints": ".keypoints",
    "KeypointDetector": ".keypoint_detector",
    "Matcher": ".matcher",
    "Templates": ".templates",
}

__all__ = [
    "ImageProcessor",
    "Image",
    "Keypoints",
    "KeypointDetector",
    "Matcher",
    "Templates",
]


def __getattr__(name: str):
    """Load public classes only when requested, preserving the existing API."""
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
