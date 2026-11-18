def assert_no_duplicates(df, keys=("trajectory_id", "image_id")):
    dup_mask = df.duplicated(subset=list(keys), keep=False)
    assert not dup_mask.any(), f"Duplicate rows for keys {keys}: {df[dup_mask]}"
def assert_flags_valid(df, flags=("interpolated", "is_last")):
    for col in flags:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique()
        assert set(vals).issubset({0, 1}), f"Invalid flag values in {col}: {vals}"
