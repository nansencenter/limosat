# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""Tests for debug recorder functionality."""

import pytest
import pandas as pd
import tempfile
import os
from datetime import datetime


def test_debug_recorder_basic():
    """Test basic debug recorder functionality."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True, run_id="test_run_123")
    
    assert recorder.enabled is True
    assert recorder.run_id == "test_run_123"
    assert recorder.get_event_count() == 0


def test_debug_recorder_record_event():
    """Test recording a basic event."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    recorder.record(
        stage="test_stage",
        event_type="info",
        message="test message",
        trajectory_id=42,
        step=10,
        test_data="value"
    )
    
    assert recorder.get_event_count() == 1
    
    df = recorder.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]['stage'] == "test_stage"
    assert df.iloc[0]['event_type'] == "info"
    assert df.iloc[0]['message'] == "test message"
    assert df.iloc[0]['trajectory_id'] == 42
    assert df.iloc[0]['step'] == 10
    assert df.iloc[0]['data_test_data'] == "value"


def test_debug_recorder_matcher_filter():
    """Test recording matcher filter events."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    recorder.record_matcher_filter(
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
        model_threshold=10000.0,
        min_homography_inliers=10,
        estimation_method="USAC_MAGSAC",
    )
    
    assert recorder.get_event_count() == 1
    
    df = recorder.to_dataframe()
    assert df.iloc[0]['stage'] == "matcher_filter"
    assert df.iloc[0]['event_type'] == "failure"
    assert df.iloc[0]['data_num_initial_matches'] == 100
    assert df.iloc[0]['data_num_descriptor_passed'] == 0


def test_debug_recorder_trajectory_termination():
    """Test recording trajectory termination events."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    recorder.record_trajectory_termination(
        trajectory_id=42,
        step=10,
        reason="no matches after filter",
        num_observations=15,
        duration_days=3.5,
    )
    
    assert recorder.get_event_count() == 1
    
    df = recorder.to_dataframe()
    assert df.iloc[0]['stage'] == "trajectory_manager"
    assert df.iloc[0]['event_type'] == "termination"
    assert df.iloc[0]['trajectory_id'] == 42
    assert df.iloc[0]['message'] == "no matches after filter"
    assert df.iloc[0]['data_num_observations'] == 15
    assert df.iloc[0]['data_duration_days'] == 3.5


def test_debug_recorder_to_feather():
    """Test writing debug events to feather file."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    # Add some events
    for i in range(5):
        recorder.record(
            stage="test_stage",
            event_type="info",
            message=f"test message {i}",
            trajectory_id=i,
            step=i * 10,
        )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_debug.feather")
        recorder.to_feather(output_path)
        
        assert os.path.exists(output_path)
        
        # Read back and verify
        df_read = pd.read_feather(output_path)
        assert len(df_read) == 5
        assert (df_read['trajectory_id'] == [0, 1, 2, 3, 4]).all()


def test_debug_recorder_get_events_for_trajectory():
    """Test filtering events by trajectory ID."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    # Add events for multiple trajectories
    for tid in [1, 2, 3]:
        for step in range(5):
            recorder.record(
                stage="test_stage",
                event_type="info",
                message=f"trajectory {tid} step {step}",
                trajectory_id=tid,
                step=step,
            )
    
    # Get events for trajectory 2
    traj2_events = recorder.get_events_for_trajectory(2)
    
    assert len(traj2_events) == 5
    assert (traj2_events['trajectory_id'] == 2).all()
    assert (traj2_events['step'] == [0, 1, 2, 3, 4]).all()


def test_debug_recorder_summary():
    """Test getting summary of recorded events."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=True)
    
    # Add diverse events
    recorder.record(stage="matcher_filter", event_type="failure", message="test", step=1)
    recorder.record(stage="matcher_filter", event_type="info", message="test", step=2)
    recorder.record(stage="pattern_match", event_type="failure", message="test", trajectory_id=1, step=3)
    recorder.record(stage="pattern_match", event_type="failure", message="test", trajectory_id=2, step=4)
    
    summary = recorder.get_summary()
    
    assert summary['total_events'] == 4
    assert summary['by_stage']['matcher_filter'] == 2
    assert summary['by_stage']['pattern_match'] == 2
    assert summary['by_type']['failure'] == 3
    assert summary['by_type']['info'] == 1
    assert summary['trajectories_tracked'] == 2


def test_noop_debug_recorder():
    """Test that NoOpDebugRecorder has minimal overhead."""
    from limosat.debug_recorder import NoOpDebugRecorder
    
    recorder = NoOpDebugRecorder()
    
    assert recorder.enabled is False
    assert recorder.run_id == "disabled"
    
    # All methods should be no-ops
    recorder.record(stage="test", event_type="info", message="test")
    recorder.record_matcher_filter(
        stage="matcher_filter",
        trajectory_id=None,
        step=5,
        event_type="failure",
        message="test",
    )
    recorder.record_trajectory_termination(
        trajectory_id=42,
        step=10,
        reason="test",
    )
    
    assert recorder.get_event_count() == 0
    
    df = recorder.to_dataframe()
    assert df.empty


def test_debug_recorder_disabled():
    """Test that disabled recorder doesn't record events."""
    from limosat.debug_recorder import DebugRecorder
    
    recorder = DebugRecorder(enabled=False)
    
    assert recorder.enabled is False
    
    recorder.record(
        stage="test_stage",
        event_type="info",
        message="test message",
        trajectory_id=42,
        step=10,
    )
    
    assert recorder.get_event_count() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
