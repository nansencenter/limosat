#!/usr/bin/env python3
"""
Example CLI-style script for running LiMOSAT with debug trajectory recording.

This script demonstrates how to integrate the debug recorder into a typical
LiMOSAT processing workflow that can be run from the command line.

Usage:
    python run_limosat_with_debug.py config.yaml

The script will:
1. Load configuration from the specified YAML file
2. Initialize debug recorder if debug.trajectories_enabled is true
3. Set up ImageProcessor with debug recording
4. Process images from the catalog
5. Write debug feather file on completion (if enabled)

Example config.yaml with debug enabled:
```yaml
run_settings:
  run_name: "my_debug_run"
  
debug:
  trajectories_enabled: true
  output_path: "./data/debug/{run_name}_debug.feather"

# ... other configuration sections ...
```
"""

import sys
import os
import yaml
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from limosat import ImageProcessor, Keypoints, Matcher, Templates
from limosat.debug_recorder import DebugRecorder, NoOpDebugRecorder
from limosat.catalog import build_stac_item_collection


def load_config(config_path):
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def initialize_debug_recorder(config):
    """
    Initialize debug recorder based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DebugRecorder or NoOpDebugRecorder instance
    """
    debug_config = config.get('debug', {})
    debug_enabled = debug_config.get('trajectories_enabled', False)
    
    if debug_enabled:
        print("🔍 Debug trajectory recording ENABLED")
        recorder = DebugRecorder(enabled=True)
        print(f"   Run ID: {recorder.run_id}")
        
        # Get output path from config
        output_path = debug_config.get('output_path', './data/debug/{run_name}_debug.feather')
        run_name = config.get('run_settings', {}).get('run_name', 'unknown')
        output_path = output_path.replace('{run_name}', run_name)
        output_path = output_path.replace('{run_id}', recorder.run_id)
        print(f"   Debug output: {output_path}")
        
        return recorder
    else:
        print("ℹ️  Debug trajectory recording DISABLED")
        return NoOpDebugRecorder()


def create_orb_model(config):
    """Create ORB feature detector from configuration."""
    kp_config = config.get('keypoint_detector', {})
    orb_params = kp_config.get('orb_params', {})
    
    # Map string score type to cv2 constant
    score_type_str = orb_params.get('scoreType', 'HARRIS_SCORE')
    score_type = getattr(cv2, score_type_str, cv2.ORB_HARRIS_SCORE)
    
    return cv2.ORB_create(
        nfeatures=orb_params.get('nfeatures', 500),
        scaleFactor=orb_params.get('scaleFactor', 1.2),
        nlevels=orb_params.get('nlevels', 8),
        edgeThreshold=orb_params.get('edgeThreshold', 31),
        firstLevel=orb_params.get('firstLevel', 0),
        patchSize=orb_params.get('patchSize', 31),
        scoreType=score_type
    )


def create_matcher(config, debug_recorder):
    """Create Matcher instance from configuration with debug recorder."""
    matcher_config = config.get('matcher_params', {})
    geom_config = matcher_config.get('geometric_model', {})
    
    # Map string norm type to cv2 constant
    norm_type_str = matcher_config.get('norm_type', 'NORM_HAMMING2')
    norm_type = getattr(cv2, norm_type_str, cv2.NORM_HAMMING2)
    
    return Matcher(
        norm=norm_type,
        descriptor_distance_max=matcher_config.get('descriptor_distance_max', 120),
        spatial_distance_max=matcher_config.get('spatial_distance_max', 100000),
        model_threshold=matcher_config.get('model_threshold', 10000),
        use_model_estimation=geom_config.get('use_model_estimation', True),
        estimation_method=geom_config.get('estimation_method', 'USAC_MAGSAC'),
        debug_recorder=debug_recorder  # Pass debug recorder to matcher
    )


def create_database_engine(config):
    """Create database engine if persistence is enabled."""
    run_settings = config.get('run_settings', {})
    if not run_settings.get('persist_updates', True):
        return None
    
    db_config = config.get('database', {})
    engine_url = db_config.get('engine_url')
    
    if not engine_url:
        print("⚠️  Warning: No database URL provided, persistence disabled")
        return None
    
    try:
        engine = create_engine(engine_url)
        print(f"✓ Database connection established")
        return engine
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to database: {e}")
        print("   Continuing without persistence")
        return None


def get_zarr_path(config):
    """Get Zarr storage path from configuration."""
    db_config = config.get('database', {})
    zarr_template = db_config.get('zarr_path_template', './data/zarr_stores/{run_name}.zarr')
    run_name = config.get('run_settings', {}).get('run_name', 'default')
    return zarr_template.replace('{run_name}', run_name)


def load_image_catalog(config):
    """Load image catalog from configuration."""
    paths_config = config.get('paths', {})
    catalog_config = paths_config.get('image_catalog', {})
    
    metadata_path = catalog_config.get('metadata_path')
    if not metadata_path or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Image catalog not found: {metadata_path}")
    
    # Load catalog
    catalog = gpd.read_file(metadata_path)
    
    # Apply date filters if specified
    start_date = catalog_config.get('start_date_filter')
    end_date = catalog_config.get('end_date_filter')
    
    if start_date:
        catalog = catalog[catalog['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        catalog = catalog[catalog['datetime'] <= pd.to_datetime(end_date)]
    
    print(f"✓ Loaded {len(catalog)} images from catalog")
    return catalog


def load_insitu_points(config):
    """Load in-situ points if configured."""
    paths_config = config.get('paths', {})
    insitu_config = paths_config.get('insitu_points', {})
    
    if not insitu_config.get('use_insitu_seeding', False):
        return None
    
    geojson_path = insitu_config.get('geojson_path')
    if not geojson_path or not os.path.exists(geojson_path):
        print(f"⚠️  Warning: In-situ data file not found: {geojson_path}")
        return None
    
    insitu_points = gpd.read_file(geojson_path)
    print(f"✓ Loaded {len(insitu_points)} in-situ points")
    return insitu_points


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_limosat_with_debug.py <config.yaml>")
        print("\nExample:")
        print("  python run_limosat_with_debug.py config.yaml")
        print("\nTo enable debug recording, add this to your config.yaml:")
        print("  debug:")
        print("    trajectories_enabled: true")
        print("    output_path: './data/debug/{run_name}_debug.feather'")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    print("=" * 70)
    print("LiMOSAT Ice Drift Tracking with Debug Recording")
    print("=" * 70)
    
    # Load configuration
    print(f"\n📋 Loading configuration from: {config_path}")
    config = load_config(config_path)
    run_name = config.get('run_settings', {}).get('run_name', 'limosat_run')
    print(f"   Run name: {run_name}")
    
    # Initialize debug recorder
    print(f"\n🔧 Initializing debug recorder...")
    debug_recorder = initialize_debug_recorder(config)
    
    # Create components
    print(f"\n🔧 Creating LiMOSAT components...")
    orb_model = create_orb_model(config)
    matcher = create_matcher(config, debug_recorder)
    
    # Database setup
    print(f"\n💾 Setting up persistence...")
    engine = create_database_engine(config)
    zarr_path = get_zarr_path(config) if engine else None
    
    # Load image catalog
    print(f"\n📷 Loading image catalog...")
    image_catalog = load_image_catalog(config)
    
    # Load in-situ points (optional)
    print(f"\n📍 Loading in-situ points...")
    insitu_points = load_insitu_points(config)
    
    # Initialize state
    points = Keypoints()
    templates = Templates()
    
    # Create ImageProcessor with debug recorder
    print(f"\n🚀 Initializing ImageProcessor with debug recording...")
    processor = ImageProcessor(
        points=points,
        model=orb_model,
        matcher=matcher,
        config=config,
        engine=engine,
        zarr_path=zarr_path,
        run_name=run_name,
        insitu_points=insitu_points,
        templates=templates,
        debug_recorder=debug_recorder,  # Pass debug recorder here
    )
    
    # Process images
    print(f"\n⚙️  Processing {len(image_catalog)} images...")
    print("-" * 70)
    
    for idx, row in image_catalog.iterrows():
        image_id = row.get('image_id', idx)
        filename = row.get('filename', row.get('filepath', 'unknown'))
        
        try:
            processor.process_image(image_id, filename)
            print(f"✓ Processed image {image_id}: {os.path.basename(filename)}")
        except Exception as e:
            print(f"✗ Error processing image {image_id}: {e}")
            if debug_recorder.enabled:
                # Record processing error
                debug_recorder.record(
                    stage="image_processing",
                    event_type="failure",
                    message=f"Error processing image: {str(e)}",
                    step=image_id,
                )
    
    # Finalize and write debug data
    print(f"\n📝 Finalizing...")
    processor.ensure_final_persistence()
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ Processing Complete!")
    print("=" * 70)
    
    if debug_recorder.enabled:
        summary = debug_recorder.get_summary()
        print(f"\n📊 Debug Recording Summary:")
        print(f"   Total events: {summary['total_events']}")
        print(f"   Trajectories tracked: {summary['trajectories_tracked']}")
        print(f"   Events by stage:")
        for stage, count in summary['by_stage'].items():
            print(f"      {stage}: {count}")
        print(f"   Events by type:")
        for event_type, count in summary['by_type'].items():
            print(f"      {event_type}: {count}")
        
        # Get output path
        debug_config = config.get('debug', {})
        output_path = debug_config.get('output_path', './data/debug/{run_name}_debug.feather')
        output_path = output_path.replace('{run_name}', run_name)
        output_path = output_path.replace('{run_id}', debug_recorder.run_id)
        print(f"\n📁 Debug data written to: {output_path}")
        print(f"\nTo analyze the debug data:")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_feather('{output_path}')")
        print(f"  print(df[df['trajectory_id'] == <your_trajectory_id>])")


if __name__ == '__main__':
    main()
