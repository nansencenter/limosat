import os
import json
import tempfile
import pytest
from datetime import datetime, timezone


@pytest.mark.unit
def test_create_stac_item_collection_empty_files():
    """Test that empty files list raises ValueError"""
    from limosat.catalog import create_stac_item_collection
    
    with pytest.raises(ValueError, match="Files list cannot be empty"):
        create_stac_item_collection([])


@pytest.mark.unit
def test_create_stac_item_collection_missing_file():
    """Test that missing file raises FileNotFoundError"""
    from limosat.catalog import create_stac_item_collection
    
    with pytest.raises(FileNotFoundError, match="File not found"):
        create_stac_item_collection(["/nonexistent/file.tiff"])


@pytest.mark.unit
def test_create_stac_item_collection_basic(tmp_path, monkeypatch):
    """Test basic STAC ItemCollection creation with valid files"""
    import limosat.catalog as catalog_module
    from limosat.catalog import create_stac_item_collection
    
    # Create mock files
    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_1234.tiff"
    file2 = tmp_path / "S1B_EW_GRDM_1SDH_20250101T120000_20250101T120030_012346_ABCDEF_5678.tiff"
    file1.touch()
    file2.touch()
    
    # Mock Image class to avoid dependencies
    class MockImage:
        def __init__(self, filepath):
            self.filepath = filepath
        
        def get_border_geojson(self):
            # Return a simple polygon in WGS84
            return '{"type": "Polygon", "coordinates": [[[-10, 70], [-9, 70], [-9, 71], [-10, 71], [-10, 70]]]}'
    
    monkeypatch.setattr(catalog_module, 'Image', MockImage)
    
    # Create catalog
    out_path = tmp_path / "catalog.json"
    result = create_stac_item_collection([str(file1), str(file2)], str(out_path))
    
    # Verify output path
    assert result == str(out_path)
    assert out_path.exists()
    
    # Load and verify catalog
    with open(out_path, 'r') as f:
        catalog = json.load(f)
    
    # Check STAC structure
    assert catalog['type'] == 'FeatureCollection'
    assert len(catalog['features']) == 2
    
    # Check first item
    item1 = catalog['features'][0]
    assert item1['id'] == '1234'  # product unique id
    assert item1['type'] == 'Feature'
    assert 'geometry' in item1
    assert 'bbox' in item1
    assert item1['properties']['image_id'] == 1
    assert item1['properties']['filename'] == file1.name
    assert item1['properties']['filepath'] == str(file1)
    assert 'image' in item1['assets']
    assert item1['assets']['image']['href'] == str(file1)
    assert item1['assets']['image']['type'] == 'image/tiff'
    
    # Check second item
    item2 = catalog['features'][1]
    assert item2['id'] == '5678'
    assert item2['properties']['image_id'] == 2


@pytest.mark.unit
def test_create_stac_item_collection_deduplication(tmp_path, monkeypatch):
    """Test that duplicate product UIDs are skipped"""
    import limosat.catalog as catalog_module
    from limosat.catalog import create_stac_item_collection
    
    # Create mock files with duplicate product UID
    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_1234.tiff"
    file2 = tmp_path / "S1B_EW_GRDM_1SDH_20250101T120000_20250101T120030_012346_ABCDEF_1234.tiff"  # Same UID
    file1.touch()
    file2.touch()
    
    # Mock Image class
    class MockImage:
        def __init__(self, filepath):
            self.filepath = filepath
        
        def get_border_geojson(self):
            return '{"type": "Polygon", "coordinates": [[[-10, 70], [-9, 70], [-9, 71], [-10, 71], [-10, 70]]]}'
    
    monkeypatch.setattr(catalog_module, 'Image', MockImage)
    
    # Create catalog
    out_path = tmp_path / "catalog.json"
    create_stac_item_collection([str(file1), str(file2)], str(out_path))
    
    # Load and verify only one item
    with open(out_path, 'r') as f:
        catalog = json.load(f)
    
    assert len(catalog['features']) == 1
    assert catalog['features'][0]['id'] == '1234'


@pytest.mark.unit
def test_create_stac_item_collection_sorting(tmp_path, monkeypatch):
    """Test that items are sorted by timestamp then filename"""
    import limosat.catalog as catalog_module
    from limosat.catalog import create_stac_item_collection
    
    # Create files with different timestamps (out of order)
    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250102T000000_20250102T000030_012345_ABCDEF_AAA1.tiff"
    file2 = tmp_path / "S1B_EW_GRDM_1SDH_20250101T000000_20250101T000030_012346_ABCDEF_BBB2.tiff"
    file3 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012347_ABCDEF_CCC3.tiff"
    file1.touch()
    file2.touch()
    file3.touch()
    
    # Mock Image class
    class MockImage:
        def __init__(self, filepath):
            self.filepath = filepath
        
        def get_border_geojson(self):
            return '{"type": "Polygon", "coordinates": [[[-10, 70], [-9, 70], [-9, 71], [-10, 71], [-10, 70]]]}'
    
    monkeypatch.setattr(catalog_module, 'Image', MockImage)
    
    # Create catalog with files out of order
    out_path = tmp_path / "catalog.json"
    create_stac_item_collection([str(file1), str(file2), str(file3)], str(out_path))
    
    # Load and verify sorting
    with open(out_path, 'r') as f:
        catalog = json.load(f)
    
    # Should be sorted: file3 (CCC3, 2025-01-01, S1A*), file2 (BBB2, 2025-01-01, S1B*), file1 (AAA1, 2025-01-02)
    # When timestamps are equal, sort by filename (S1A comes before S1B)
    assert len(catalog['features']) == 3
    assert catalog['features'][0]['id'] == 'CCC3'
    assert catalog['features'][0]['properties']['image_id'] == 1
    assert catalog['features'][1]['id'] == 'BBB2'
    assert catalog['features'][1]['properties']['image_id'] == 2
    assert catalog['features'][2]['id'] == 'AAA1'
    assert catalog['features'][2]['properties']['image_id'] == 3


@pytest.mark.unit
def test_create_stac_item_collection_atomic_write(tmp_path, monkeypatch):
    """Test that output is written atomically"""
    import limosat.catalog as catalog_module
    from limosat.catalog import create_stac_item_collection
    
    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_1234.tiff"
    file1.touch()
    
    # Mock Image class
    class MockImage:
        def __init__(self, filepath):
            self.filepath = filepath
        
        def get_border_geojson(self):
            return '{"type": "Polygon", "coordinates": [[[-10, 70], [-9, 70], [-9, 71], [-10, 71], [-10, 70]]]}'
    
    monkeypatch.setattr(catalog_module, 'Image', MockImage)
    
    # Create catalog
    out_path = tmp_path / "catalog.json"
    create_stac_item_collection([str(file1)], str(out_path))
    
    # Verify temp file is cleaned up
    tmp_file = tmp_path / "catalog.json.tmp"
    assert not tmp_file.exists()
    assert out_path.exists()
