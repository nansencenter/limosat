# Minimal LiMOSAT run

The previous notebook duplicated deployment setup, used outdated catalog and
template APIs, and enabled persistence without configuring storage. This example
shows the current library-level workflow without persistence.

```python
from pathlib import Path

import cv2

from limosat import ImageProcessor, Keypoints, Matcher, Templates
from limosat.catalog import build_stac_item_collection


image_paths = sorted(Path("/path/to/analysis-ready-images").glob("*.tiff"))
catalog = build_stac_item_collection(image_paths, check_exists=True)

model = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    patchSize=31,
    scoreType=cv2.ORB_HARRIS_SCORE,
)
processor = ImageProcessor(
    points=Keypoints(),
    templates=Templates(),
    model=model,
    matcher=Matcher(
        norm=cv2.NORM_HAMMING2,
        descriptor_distance_max=120,
        spatial_distance_max=100_000,
        model_threshold=15_000,
    ),
    persist_updates=False,
)

for item in catalog.items:
    processor.process_image(
        image_id=item.properties["image_id"],
        filename=item.assets["image"].href,
    )

points = processor.points
```

Input images must already satisfy the band and georeferencing requirements in
the main README. `build_stac_item_collection` currently recognizes LiMOSAT's
Sentinel-1 filename convention and assigns deterministic image IDs in acquisition
order.

Production runs should supply their validated deployment configuration, SQL
engine, Zarr path, and run name rather than copying those details into this
library example.
