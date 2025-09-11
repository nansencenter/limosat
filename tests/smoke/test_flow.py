import types
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from tests.factories import ImageStub, MatcherStub, DatabaseStub, make_templates


@pytest.mark.smoke
def test_process_image_smoke(monkeypatch):
    # Build minimal Keypoints and Templates using simple GeoDataFrame-compatible structure
    from limosat.keypoints import Keypoints
    from limosat.templates import Templates
    from limosat.image_processor import ImageProcessor

    points = Keypoints()
    templates = Templates()

    # Construct processor with stubs
    matcher = MatcherStub()
    proc = ImageProcessor(points=points, model=None, matcher=matcher, persist_updates=True)

    # Monkeypatch DB with stub
    db_stub = DatabaseStub()
    monkeypatch.setattr(proc, 'db', db_stub, raising=False)

    # Provide an ImageStub and call process_image
    img = ImageStub()

    # Monkeypatch constructor of limosat.image.Image to return our stub
    import limosat.image as image_mod
    monkeypatch.setattr(image_mod, 'Image', lambda filename: img)

    # Run twice for idempotence-lite
    proc.process_image(image_id=1, filename='dummy')
    proc.process_image(image_id=1, filename='dummy')

    # Assertions
    assert db_stub.calls <= 2  # called at most once per invocation
