from netCDF4 import Dataset
import numpy as np
import pandas as pd

from experiments.validate_cryosat2_deformation import (
    add_surface_class,
    aggregate_track_bins,
    cryosat_quality_mask,
    cryosat_utc,
    load_rdwes1b,
)


def test_cryosat_epoch_conversion_is_utc():
    result = cryosat_utc(np.array([0.0, 1.0]), np.array([0.0, 60.0]))
    assert result[0] == pd.Timestamp("2000-01-01T00:00:00Z")
    assert result[1] == pd.Timestamp("2000-01-02T00:01:00Z")


def test_quality_rule_retains_fit_bound_only_in_primary_analysis():
    observations = pd.DataFrame(
        {
            "roughness_m": [0.2, 1.0, 0.3, -0.1],
            "norm_res": [0.1, 0.2, 0.6, 0.1],
        }
    )
    assert cryosat_quality_mask(observations).tolist() == [True, True, False, False]
    assert cryosat_quality_mask(
        observations, exclude_one_metre_fit_bound=True
    ).tolist() == [True, False, False, False]


def test_surface_classes_use_published_sar_mode_thresholds():
    observations = pd.DataFrame(
        {
            "peakiness": [0.19, 0.08, 0.12],
            "stack_sd": [3.9, 4.1, 4.0],
            "roughness_m": [0.01, 0.3, 0.2],
        }
    )

    result = add_surface_class(observations)

    assert result["surface_class"].tolist() == ["lead", "floe", "ambiguous"]
    assert result["lead_return"].tolist() == [1.0, 0.0, 0.0]
    assert result["floe_return"].tolist() == [0.0, 1.0, 0.0]
    assert np.isnan(result.loc[0, "floe_roughness_m"])
    assert result.loc[1, "floe_roughness_m"] == 0.3


def test_loader_projects_longitude_and_preserves_physical_units(tmp_path):
    path = tmp_path / "RDWES1B_fixture.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 3)
        values = {
            "lat": [80.0, 80.01, 80.02],
            "lon": [350.0, 350.0, 350.0],
            "elev": [10.0, 11.0, 12.0],
            "retrack_elev": [0.1, 0.2, 0.3],
            "roughness": [0.2, 0.3, 0.4],
            "norm_res": [0.1, 0.2, 0.3],
            "peakiness": [0.01, 0.02, 0.03],
            "stack_sd": [1.0, 2.0, 3.0],
            "day": [1.0, 1.0, 1.0],
            "sec": [0.0, 1.0, 2.0],
            "i": [0, 1, 2],
        }
        for name, data in values.items():
            dtype = "i4" if name == "i" else "f8"
            dataset.createVariable(name, dtype, ("x",))[:] = data

    result = load_rdwes1b([path])

    assert len(result) == 3
    np.testing.assert_allclose(result["longitude"], -10.0)
    np.testing.assert_allclose(result["surface_elevation_m"], [10.1, 11.2, 12.3])
    assert result["along_track_m"].is_monotonic_increasing
    assert result.loc[0, "time_utc"] == pd.Timestamp("2000-01-02T00:00:00Z")


def test_track_bins_use_medians_and_minimum_footprint_count():
    observations = pd.DataFrame(
        {
            "track_id": ["track"] * 5,
            "along_track_m": [0.0, 100.0, 200.0, 1100.0, 1200.0],
            "roughness_m": [0.1, 0.3, 0.2, 0.8, 0.9],
            "lead_return": [1.0, 0.0, 0.0, 0.0, 0.0],
            "floe_return": [0.0, 1.0, 1.0, 1.0, 1.0],
            "floe_roughness_m": [np.nan, 0.3, 0.2, 0.8, 0.9],
            "norm_res": 0.1,
            "peakiness": 0.2,
            "stack_sd": 2.0,
            "laser_x": np.arange(5, dtype=float),
            "laser_y": np.arange(5, dtype=float),
            "longitude": -20.0,
            "latitude": 80.0,
            "time_utc": pd.date_range("2020-01-01", periods=5, tz="UTC"),
            "test_available": True,
            **{
                f"test_{name}": np.arange(5, dtype=float)
                for name in (
                    "divergence_per_day",
                    "shear_per_day",
                    "total_per_day",
                    "maximum_compression_per_day",
                    "maximum_extension_per_day",
                )
            },
        }
    )

    result = aggregate_track_bins(
        observations, "test", bin_size_m=1000.0, minimum_footprints=3
    )

    assert len(result) == 1
    assert result.loc[0, "footprints"] == 3
    assert np.isclose(result.loc[0, "roughness_m"], 0.2)
