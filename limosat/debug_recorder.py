# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
Debug recorder for trajectory and matching failure analysis.

This module provides infrastructure for recording debug events during limosat runs,
enabling detailed analysis of trajectory termination and matching failures.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
from .utils import logger


@dataclass
class DebugEvent:
    """
    Structured debug event for trajectory analysis.
    
    Attributes:
        run_id: Unique identifier for the debug run
        stage: Pipeline stage where event occurred (e.g., "bf_matcher", "matcher_filter", "keypoint_detector")
        event_type: Type of event (e.g., "info", "warning", "failure")
        message: Human-readable description of the event
        trajectory_id: Optional trajectory identifier
        step: Optional step/frame/image_id
        timestamp: Event timestamp
        data: Flexible dictionary for stage-specific metrics
    """
    run_id: str
    stage: str
    event_type: str
    message: str
    trajectory_id: Optional[int] = None
    step: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to flat dictionary suitable for DataFrame."""
        result = {
            'run_id': self.run_id,
            'stage': self.stage,
            'event_type': self.event_type,
            'message': self.message,
            'trajectory_id': self.trajectory_id,
            'step': self.step,
            'timestamp': self.timestamp,
        }
        # Flatten data dict into result with prefix
        for key, value in self.data.items():
            # Convert numpy types to native Python types for feather compatibility
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            result[f'data_{key}'] = value
        return result


class DebugRecorder:
    """
    Central recorder for debug events during a limosat run.
    
    This class provides:
    - In-memory event accumulation for smaller debug runs
    - Structured event recording with flexible data fields
    - Export to pandas DataFrame and feather format
    - Run-level unique identifiers
    """
    
    def __init__(self, enabled: bool = False, run_id: Optional[str] = None):
        """
        Initialize debug recorder.
        
        Args:
            enabled: Whether debug recording is active
            run_id: Optional run identifier (generated if not provided)
        """
        self.enabled = enabled
        self.run_id = run_id or self._generate_run_id()
        self._events: List[DebugEvent] = []
        
        if self.enabled:
            logger.info(f"Debug recording enabled with run_id: {self.run_id}")

    def __bool__(self) -> bool:
        """Allow truthiness checks to follow enabled state."""
        return bool(self.enabled)
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"debug_{timestamp}_{short_uuid}"
    
    def record(
        self,
        stage: str,
        event_type: str,
        message: str,
        trajectory_id: Optional[int] = None,
        step: Optional[int] = None,
        **data
    ):
        """
        Record a debug event.
        
        Args:
            stage: Pipeline stage (e.g., "matcher_filter", "bf_matcher")
            event_type: Event type (e.g., "info", "warning", "failure")
            message: Human-readable event description
            trajectory_id: Optional trajectory identifier
            step: Optional step/frame/image_id
            **data: Additional stage-specific data as keyword arguments
        """
        if not self.enabled:
            return
        
        event = DebugEvent(
            run_id=self.run_id,
            stage=stage,
            event_type=event_type,
            message=message,
            trajectory_id=trajectory_id,
            step=step,
            data=data
        )
        self._events.append(event)
    
    def record_matcher_filter(
        self,
        stage: str,
        trajectory_id: Optional[int],
        step: Optional[int],
        event_type: str,
        message: str,
        num_initial_matches: int = 0,
        num_descriptor_passed: int = 0,
        num_spatial_passed: int = 0,
        num_homography_inliers: int = 0,
        descriptor_distance_max: Optional[float] = None,
        spatial_distance_max: Optional[float] = None,
        model_threshold: Optional[float] = None,
        min_homography_inliers: Optional[int] = None,
        estimation_method: Optional[str] = None,
        inlier_ratio: Optional[float] = None,
        residual_median: Optional[float] = None,
        residual_mean: Optional[float] = None,
        num_traj_descriptor: Optional[int] = None,
        num_traj_spatial: Optional[int] = None,
        num_traj_inliers: Optional[int] = None,
        trajectory_ids_sample: Optional[List[int]] = None,
    ):
        """
        Record a matcher filter event with standardized fields.
        
        This is a convenience method for recording detailed matcher filter events.
        """
        self.record(
            stage=stage,
            event_type=event_type,
            message=message,
            trajectory_id=trajectory_id,
            step=step,
            num_initial_matches=num_initial_matches,
            num_descriptor_passed=num_descriptor_passed,
            num_spatial_passed=num_spatial_passed,
            num_homography_inliers=num_homography_inliers,
            descriptor_distance_max=descriptor_distance_max,
            spatial_distance_max=spatial_distance_max,
            model_threshold=model_threshold,
            min_homography_inliers=min_homography_inliers,
            estimation_method=estimation_method,
            inlier_ratio=inlier_ratio,
            residual_median=residual_median,
            residual_mean=residual_mean,
            num_traj_descriptor=num_traj_descriptor,
            num_traj_spatial=num_traj_spatial,
            num_traj_inliers=num_traj_inliers,
            trajectory_ids_sample=trajectory_ids_sample,
        )
    
    def record_keypoint_detection(
        self,
        stage: str,
        trajectory_id: Optional[int],
        step: Optional[int],
        keypoint_x: float,
        keypoint_y: float,
        response: float,
        composite_score: Optional[float] = None,
    ):
        """
        Record a keypoint detection event with response values.
        """
        self.record(
            stage=stage,
            event_type="info",
            message="keypoint_detected",
            trajectory_id=trajectory_id,
            step=step,
            keypoint_x=keypoint_x,
            keypoint_y=keypoint_y,
            response=response,
            composite_score=composite_score,
        )
    
    def record_trajectory_termination(
        self,
        trajectory_id: int,
        step: int,
        reason: str,
        num_observations: int = 0,
        duration_days: Optional[float] = None,
        **additional_data
    ):
        """
        Record a trajectory termination event.
        
        Args:
            trajectory_id: Trajectory identifier
            step: Image/frame where termination occurred
            reason: Human-readable termination reason
            num_observations: Number of observations in trajectory
            duration_days: Duration of trajectory in days
            **additional_data: Additional trajectory statistics
        """
        self.record(
            stage="trajectory_manager",
            event_type="termination",
            message=reason,
            trajectory_id=trajectory_id,
            step=step,
            num_observations=num_observations,
            duration_days=duration_days,
            **additional_data
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all recorded events to a pandas DataFrame.
        
        Returns:
            DataFrame with one row per event
        """
        if not self._events:
            return pd.DataFrame()
        
        rows = [event.to_dict() for event in self._events]
        return pd.DataFrame(rows)
    
    def to_feather(self, path: str):
        """
        Write all recorded events to a feather file.
        
        Args:
            path: Output path for feather file (will append .feather if not present)
        """
        df = self.to_dataframe()
        if df.empty:
            logger.warning(f"No debug events to write to {path}")
            return
        
        # Ensure the path ends with .feather extension
        if not path.endswith('.feather'):
            logger.info(f"Appending .feather extension to path: {path}")
            path = f"{path}.feather"
        
        df.to_feather(path)
        logger.info(f"Wrote {len(df)} debug events to {path}")
    
    def get_events_for_trajectory(self, trajectory_id: int) -> pd.DataFrame:
        """
        Get all events for a specific trajectory.
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            DataFrame filtered to the specified trajectory, sorted by step/timestamp
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        
        trajectory_events = df[df['trajectory_id'] == trajectory_id].copy()
        
        # Sort by step (if available), then timestamp
        if 'step' in trajectory_events.columns and not trajectory_events['step'].isna().all():
            trajectory_events = trajectory_events.sort_values(['step', 'timestamp'])
        else:
            trajectory_events = trajectory_events.sort_values('timestamp')
        
        return trajectory_events
    
    def get_event_count(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recorded events.
        
        Returns:
            Dictionary with event counts by stage and type
        """
        if not self._events:
            return {
                'total_events': 0,
                'by_stage': {},
                'by_type': {},
                'trajectories_tracked': 0,
            }
        
        df = self.to_dataframe()
        
        return {
            'total_events': len(df),
            'by_stage': df['stage'].value_counts().to_dict(),
            'by_type': df['event_type'].value_counts().to_dict(),
            'trajectories_tracked': df['trajectory_id'].nunique() if 'trajectory_id' in df.columns else 0,
        }


class NoOpDebugRecorder(DebugRecorder):
    """
    No-op debug recorder for when debug mode is disabled.
    
    Provides the same interface but does nothing, ensuring minimal overhead.
    """
    
    def __init__(self):
        # Don't call parent __init__ to avoid any initialization overhead
        self.enabled = False
        self.run_id = "disabled"
        self._events = []
    
    def record(self, *args, **kwargs):
        """No-op record method."""
        pass
    
    def record_matcher_filter(self, *args, **kwargs):
        """No-op record method."""
        pass
    
    def record_keypoint_detection(self, *args, **kwargs):
        """No-op record method."""
        pass
    
    def record_trajectory_termination(self, *args, **kwargs):
        """No-op record method."""
        pass
    
    def to_feather(self, path: str):
        """No-op write method."""
        pass
