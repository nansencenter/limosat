# Debug Trajectory Infrastructure

## Overview

The debug trajectory infrastructure provides detailed recording and analysis capabilities for understanding why trajectories terminate or matching fails in LiMOSAT. This feature is designed for smaller, targeted debug runs to diagnose issues in the ice drift tracking pipeline.

## Key Features

### 1. **Structured Event Recording**
- Records events at key pipeline stages (matching, filtering, pattern matching, trajectory management)
- Captures relevant parameters and failure modes
- Associates events with specific trajectory IDs for trajectory-centric analysis

### 2. **Minimal Overhead When Disabled**
- Uses a no-op recorder when debug mode is disabled
- Zero performance impact on production runs

### 3. **Feather File Output**
- All debug events written to a single feather file per run
- Efficient storage and fast loading with pandas
- Easy to analyze with standard data science tools

### 4. **Trajectory-Centric Analysis**
- Query all events for a specific trajectory
- Identify the exact stage and reason where each trajectory terminated
- Track keypoint responses and other metrics over trajectory lifetime

## Configuration

Add the debug section to your `config.yaml`:

```yaml
debug:
  trajectories_enabled: true  # Enable debug recording
  output_path: "./data/debug/{run_name}_debug.feather"  # Output path (supports {run_name} and {run_id} placeholders)
```

## Usage

### CLI / Script Usage

If you have your own run script (e.g., `run_limosat.py`), you need to:

1. **Add debug config to your YAML file:**
   ```yaml
   debug:
     trajectories_enabled: true
     output_path: "./data/debug/{run_name}_debug.feather"
   ```

2. **Load config and create debug recorder in your script:**
   ```python
   import yaml
   from limosat.debug_recorder import DebugRecorder, NoOpDebugRecorder
   
   # Load your config
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Create debug recorder based on config
   debug_enabled = config.get('debug', {}).get('trajectories_enabled', False)
   if debug_enabled:
       debug_recorder = DebugRecorder(enabled=True)
   else:
       debug_recorder = NoOpDebugRecorder()
   ```

3. **Pass debug_recorder to Matcher and ImageProcessor:**
   ```python
   # Create matcher WITH debug recorder
   matcher = Matcher(
       norm=cv2.NORM_HAMMING2,
       descriptor_distance_max=120,
       # ... other params ...
       debug_recorder=debug_recorder  # Add this
   )
   
   # Create processor WITH debug recorder
   processor = ImageProcessor(
       points=points,
       model=orb_model,
       matcher=matcher,
       config=config,  # Pass config so it can access debug.output_path
       debug_recorder=debug_recorder,  # Add this
       # ... other parameters ...
   )
   ```

4. **Run your script normally:**
   ```bash
   python run_limosat.py config.yaml
   
   # Or with nohup:
   nohup python run_limosat.py config.yaml > run_debug.log 2>&1 &
   ```

**Complete example script:** See `examples/run_limosat_with_debug.py` for a full working example.

### Programmatic Usage

```python
from limosat.debug_recorder import DebugRecorder, NoOpDebugRecorder
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create debug recorder
debug_enabled = config.get('debug', {}).get('trajectories_enabled', False)
if debug_enabled:
    debug_recorder = DebugRecorder(enabled=True)
else:
    debug_recorder = NoOpDebugRecorder()

# Pass to ImageProcessor
from limosat.image_processor import ImageProcessor
processor = ImageProcessor(
    points=keypoints,
    model=orb_model,
    matcher=matcher,
    config=config,
    debug_recorder=debug_recorder,  # Pass debug recorder
    # ... other parameters
)

# Run your processing...

# Finalize (writes feather file)
processor.ensure_final_persistence()
```

### Analyzing Debug Data

```python
import pandas as pd

# Load debug feather file
df = pd.read_feather('./data/debug/my_run_debug.feather')

# View all events
print(df)

# Filter by trajectory ID
traj_42_events = df[df['trajectory_id'] == 42]
print(traj_42_events)

# Find termination reasons
terminations = df[df['event_type'] == 'termination']
print(terminations[['trajectory_id', 'message', 'data_num_observations']])

# Analyze failure modes
failures = df[df['event_type'] == 'failure']
print(failures.groupby('stage')['message'].value_counts())

# Check keypoint responses (if recorded)
if 'data_response' in df.columns:
    print(df['data_response'].describe())
```

## Recorded Events

### Matcher Filter Stage (`matcher_filter`)

Records filtering at descriptor, spatial, and homography stages:

- **Fields recorded:**
  - `num_initial_matches`: Initial number of matches
  - `num_descriptor_passed`: Matches passing descriptor distance filter
  - `num_spatial_passed`: Matches passing spatial distance filter
  - `num_homography_inliers`: Inliers after homography estimation
  - `descriptor_distance_max`, `spatial_distance_max`, `model_threshold`: Threshold parameters
  - `estimation_method`: Homography estimation method used

- **Failure modes recorded:**
  - "no matches passed descriptor distance filter"
  - "no matches passed spatial distance filter"
  - "insufficient matches for model estimation"
  - "model estimation failed / H is None"
  - "not enough inliers after homography estimation"
  - OpenCV errors during model estimation

### BF Matcher Stage (`bf_matcher`)

Records matching statistics:

- **Fields recorded:**
  - `num_crosscheck_matches`: Matches from BF cross-check
  - `num_lowe_additional`: Additional matches from Lowe's ratio test
  - `num_total_candidates`: Total candidate matches
  - `num_groups`: Number of match groups

- **Failure modes recorded:**
  - "no valid inlier groups found after filtering"

### Pattern Matching Stage (`pattern_match`)

Records template matching results:

- **Fields recorded:**
  - `correlation`: Template matching correlation score
  - `min_correlation`: Minimum correlation threshold

- **Failure modes recorded:**
  - "correlation below threshold"

### Keypoint Detection Stage (`keypoint_detector`)

Records keypoint properties:

- **Fields recorded:**
  - `keypoint_x`, `keypoint_y`: Keypoint location
  - `response`: Raw detector response value
  - `composite_score`: Weighted score (if window-based weighting used)

### Trajectory Manager Stage (`trajectory_manager`)

Records trajectory lifecycle events:

- **Event type:** `termination`
- **Fields recorded:**
  - `trajectory_id`: Trajectory identifier
  - `num_observations`: Number of observations in trajectory
  - `duration_days`: Duration of trajectory in days
  - Additional context-specific fields (e.g., `converged_to`)

- **Termination reasons recorded:**
  - "trajectory converged to winner trajectory X"
  - "no matches found after descriptor matching and filtering"
  - "correlation below threshold"
  - Custom reasons from other pipeline stages

## Debug Event Schema

Each debug event contains:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Unique identifier for the debug run |
| `stage` | str | Pipeline stage (e.g., "matcher_filter", "pattern_match") |
| `event_type` | str | Event type (e.g., "info", "warning", "failure", "termination") |
| `message` | str | Human-readable event description |
| `trajectory_id` | int (nullable) | Trajectory identifier (if applicable) |
| `step` | int (nullable) | Step/frame/image_id (if applicable) |
| `timestamp` | datetime | Event timestamp |
| `data_*` | various | Stage-specific metrics (flattened from data dict) |

## Example Workflow

### 1. Enable Debug Mode

Edit `config.yaml`:
```yaml
debug:
  trajectories_enabled: true
  output_path: "./data/debug/{run_name}_debug.feather"
```

### 2. Run Processing

Run your LiMOSAT processing as normal. The debug recorder will capture events automatically.

### 3. Analyze Results

```python
import pandas as pd

# Load debug data
df = pd.read_feather('./data/debug/my_run_debug.feather')

# Example 1: Find why trajectory 123 ended
traj_events = df[df['trajectory_id'] == 123].sort_values('step')
print(traj_events[['step', 'stage', 'event_type', 'message']].tail(5))

# Example 2: Analyze matcher filter success rates
filter_events = df[df['stage'] == 'matcher_filter']
success_rate = (filter_events['event_type'] == 'info').mean()
print(f"Matcher filter success rate: {success_rate:.2%}")

# Example 3: Common failure modes
failures = df[df['event_type'] == 'failure']
print(failures.groupby(['stage', 'message']).size().sort_values(ascending=False))

# Example 4: Trajectory statistics
terminations = df[df['event_type'] == 'termination']
print(f"Mean trajectory duration: {terminations['data_duration_days'].mean():.2f} days")
print(f"Mean observations per trajectory: {terminations['data_num_observations'].mean():.1f}")
```

## Performance Considerations

- **Enabled:** Events are stored in memory during the run. For large runs, this may use significant RAM. Recommended for debug runs with <1000 images.
- **Disabled:** Uses `NoOpDebugRecorder` with zero overhead (all recording methods are no-ops).

## Advanced Usage

### Custom Events

You can record custom debug events:

```python
debug_recorder.record(
    stage="my_custom_stage",
    event_type="info",
    message="custom event message",
    trajectory_id=42,
    step=10,
    custom_metric=123.45,
    custom_flag=True,
)
```

### Trajectory-Centric Queries

```python
# Get all events for a specific trajectory
traj_events = debug_recorder.get_events_for_trajectory(42)

# Or from the feather file
df = pd.read_feather('debug.feather')
traj_events = df[df['trajectory_id'] == 42].sort_values(['step', 'timestamp'])
```

### Summary Statistics

```python
summary = debug_recorder.get_summary()
print(f"Total events: {summary['total_events']}")
print(f"Events by stage: {summary['by_stage']}")
print(f"Events by type: {summary['by_type']}")
print(f"Trajectories tracked: {summary['trajectories_tracked']}")
```

## Implementation Details

### Components

- **`DebugRecorder`**: Main recorder class with in-memory event storage
- **`NoOpDebugRecorder`**: Zero-overhead no-op implementation for disabled mode
- **`DebugEvent`**: Dataclass representing a single debug event

### Integration Points

The debug recorder is integrated at:

1. **Matcher class** (`matcher.py`):
   - `filter()` method: Records filtering stages and failure modes
   - `match_with_grid()`: Records matching statistics

2. **KeypointDetector class** (`keypoint_detector.py`):
   - `detect_new_keypoints()`: Stores keypoint responses in tags
   - `keypoint_from_point()`: Stores responses for in-situ seeded points

3. **ImageProcessor class** (`image_processor.py`):
   - `_handle_trajectory_convergence()`: Records convergence terminations
   - `_match_existing_points()`: Records matching failures
   - `ensure_final_persistence()`: Writes feather file

### File Format

Debug data is stored in Apache Arrow Feather format:
- Fast to write and read
- Efficient columnar storage
- Native pandas integration
- Cross-language compatibility

## Troubleshooting

### No debug file generated
- Check that `debug.trajectories_enabled: true` in your config
- Verify the output path is writable
- Check logs for error messages during feather write

### Large memory usage
- Debug mode stores all events in memory
- For large runs, consider:
  - Running smaller subsets for debugging
  - Disabling debug mode for production runs
  - Filtering events by trajectory_id after loading

### Missing trajectory termination events
- Some trajectories may not have explicit termination events if they're simply not matched in subsequent frames
- Check for implicit terminations by finding trajectories that don't appear in later steps

## See Also

- `examples/debug_example.py`: Full example script
- `limosat/debug_recorder.py`: Implementation
- `tests/unit/test_debug_recorder.py`: Unit tests
