#!/usr/bin/env python3
"""
Example script demonstrating the debug trajectory recording feature.

This script shows how to:
1. Enable debug trajectory recording via configuration
2. Initialize the debug recorder
3. Pass it to ImageProcessor and Matcher
4. Write debug data to a feather file

Usage:
    python examples/debug_example.py

The debug feather file will be written to ./data/debug/<run_name>_debug.feather
"""

import yaml
from limosat.debug_recorder import DebugRecorder, NoOpDebugRecorder


def load_config_with_debug(config_path='config.yaml'):
    """
    Load configuration and check if debug mode is enabled.
    
    Returns:
        tuple: (config dict, debug_recorder instance)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if debug trajectories is enabled
    debug_config = config.get('debug', {})
    debug_enabled = debug_config.get('trajectories_enabled', False)
    debug_output_path = debug_config.get('output_path', './data/debug/{run_name}_debug.feather')
    
    if debug_enabled:
        print(f"🔍 Debug mode ENABLED")
        recorder = DebugRecorder(enabled=True)
        print(f"   Run ID: {recorder.run_id}")
        print(f"   Output path template: {debug_output_path}")
    else:
        print(f"ℹ️  Debug mode DISABLED (minimal overhead)")
        recorder = NoOpDebugRecorder()
    
    return config, recorder, debug_output_path


def initialize_components_with_debug(config, debug_recorder):
    """
    Initialize ImageProcessor, Matcher, and other components with debug recorder.
    
    This shows the integration pattern for adding debug recording to the pipeline.
    """
    # Example: Initialize Matcher with debug recorder
    from limosat.matcher import Matcher
    import cv2
    
    matcher_params = config.get('matcher_params', {})
    matcher = Matcher(
        norm=getattr(cv2, matcher_params.get('norm_type', 'NORM_HAMMING2')),
        descriptor_distance_max=matcher_params.get('descriptor_distance_max', 120),
        spatial_distance_max=matcher_params.get('spatial_distance_max', 100000),
        model_threshold=matcher_params.get('model_threshold', 10000),
        use_model_estimation=matcher_params.get('geometric_model', {}).get('use_model_estimation', True),
        estimation_method=matcher_params.get('geometric_model', {}).get('estimation_method', 'USAC_MAGSAC'),
        debug_recorder=debug_recorder  # Pass debug recorder
    )
    
    # Example: Initialize ImageProcessor with debug recorder
    # (Pseudo-code - actual initialization requires more setup)
    # from limosat.image_processor import ImageProcessor
    # processor = ImageProcessor(
    #     points=keypoints,
    #     model=orb_model,
    #     matcher=matcher,
    #     config=config,
    #     debug_recorder=debug_recorder,  # Pass debug recorder
    #     ...
    # )
    
    return matcher


def finalize_debug_recording(debug_recorder, output_path, run_name='example_run'):
    """
    Finalize debug recording and write feather file.
    
    This is typically called at the end of ImageProcessor.ensure_final_persistence()
    """
    if not debug_recorder.enabled:
        print("Debug recording disabled, no output written.")
        return
    
    # Process path template
    import os
    output_path = output_path.replace('{run_name}', run_name)
    output_path = output_path.replace('{run_id}', debug_recorder.run_id)
    
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write feather file
    debug_recorder.to_feather(output_path)
    
    # Print summary
    summary = debug_recorder.get_summary()
    print(f"\n📊 Debug Summary:")
    print(f"   Total events: {summary['total_events']}")
    print(f"   Trajectories tracked: {summary['trajectories_tracked']}")
    print(f"   Events by stage: {summary['by_stage']}")
    print(f"   Events by type: {summary['by_type']}")
    print(f"\n📁 Debug data written to: {output_path}")


def example_usage():
    """
    Example workflow showing how to use debug recording.
    """
    print("=" * 60)
    print("LiMOSAT Debug Trajectory Recording Example")
    print("=" * 60)
    
    # Note: This example assumes you have a config.yaml file
    # For testing, you can use config.defaults.yaml as a starting point
    
    # 1. Load configuration and create debug recorder
    try:
        config, debug_recorder, debug_output_path = load_config_with_debug('config.defaults.yaml')
    except FileNotFoundError:
        print("Warning: config.yaml not found, using default configuration")
        config = {'debug': {'trajectories_enabled': True}}
        debug_recorder = DebugRecorder(enabled=True)
        debug_output_path = './data/debug/{run_name}_debug.feather'
    
    # 2. Record some example events
    if debug_recorder.enabled:
        print("\n📝 Recording example debug events...")
        
        # Example: Matcher filter failure
        debug_recorder.record_matcher_filter(
            stage="matcher_filter",
            trajectory_id=None,
            step=5,
            event_type="failure",
            message="no matches passed descriptor distance filter",
            num_initial_matches=100,
            num_descriptor_passed=0,
            num_spatial_passed=0,
            num_homography_inliers=0,
            descriptor_distance_max=120.0,
            spatial_distance_max=100000.0,
        )
        
        # Example: Trajectory termination
        debug_recorder.record_trajectory_termination(
            trajectory_id=42,
            step=10,
            reason="no matches found after descriptor matching",
            num_observations=15,
            duration_days=3.5,
        )
        
        # Example: Pattern matching failure
        debug_recorder.record(
            stage="pattern_match",
            event_type="failure",
            message="correlation below threshold",
            trajectory_id=42,
            step=11,
            correlation=0.25,
            min_correlation=0.35,
        )
        
        print(f"   Recorded {debug_recorder.get_event_count()} events")
    
    # 3. Finalize and write debug file
    finalize_debug_recording(debug_recorder, debug_output_path, run_name='example_run')
    
    # 4. Example: Query events for a specific trajectory
    if debug_recorder.enabled and debug_recorder.get_event_count() > 0:
        print(f"\n🔍 Events for trajectory 42:")
        traj_events = debug_recorder.get_events_for_trajectory(42)
        if not traj_events.empty:
            for idx, row in traj_events.iterrows():
                print(f"   Step {row['step']}: {row['stage']} - {row['message']}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    example_usage()
