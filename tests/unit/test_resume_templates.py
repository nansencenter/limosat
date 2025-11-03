import numpy as np
import pandas as pd
import pytest
from limosat.keypoints import Keypoints
from limosat.templates import Templates
from limosat.image_processor import ImageProcessor
from tests.factories import make_templates, make_keypoints


def test_image_processor_accepts_templates():
    """Test that ImageProcessor accepts and uses provided templates."""
    points_gdf = make_keypoints(n=3, image_id=0, t0='2025-01-01 00:00:00')
    points = Keypoints._from_gdf(points_gdf)
    
    templates = Templates()
    templates.data = make_templates(tids=[0, 1, 2], hs=16)
    templates._initialized = True
    
    proc = ImageProcessor(
        points=points,
        model=None,
        matcher=None,
        templates=templates,
        persist_updates=False
    )
    
    assert len(proc.templates) == 3
    assert np.array_equal(proc.templates.trajectory_ids, [0, 1, 2])


def test_image_processor_defaults_to_empty_templates():
    """Test that ImageProcessor creates empty templates when none provided."""
    points = Keypoints()
    
    proc = ImageProcessor(
        points=points,
        model=None,
        matcher=None,
        persist_updates=False
    )
    
    assert len(proc.templates) == 0
    assert isinstance(proc.templates, Templates)


def test_resume_preserves_templates():
    """Test that templates loaded during resume are preserved."""
    points_gdf = make_keypoints(n=5, image_id=10, t0='2025-01-01 00:00:00')
    points = Keypoints._from_gdf(points_gdf)
    
    templates = Templates()
    templates.data = make_templates(tids=[0, 1, 2, 3, 4], hs=16)
    templates._initialized = True
    
    proc = ImageProcessor(
        points=points,
        model=None,
        matcher=None,
        templates=templates,
        persist_updates=False
    )
    
    assert len(proc.templates) == 5
    assert proc.templates is templates
    assert np.array_equal(proc.templates.trajectory_ids, [0, 1, 2, 3, 4])
