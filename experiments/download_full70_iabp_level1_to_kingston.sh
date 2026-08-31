#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS="${TARGETS:-$ROOT_DIR/results/iabp_s1_stratified_coverage/full70_iabp_level1_targets.csv}"
OUTPUT_ROOT="/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020/iabp_level1_full70"
SEED_ROOT="/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020/iabp_level1_selected"

if [[ ! -d /Volumes/KINGSTON ]]; then
  echo "KINGSTON is not mounted." >&2
  exit 2
fi
if [[ ! -f "$TARGETS" ]]; then
  echo "IABP target manifest is missing: $TARGETS" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

# Reuse the eight already verified official downloads and the official summary.
if [[ -d "$SEED_ROOT" ]]; then
  for seed_file in "$SEED_ROOT"/*.csv "$SEED_ROOT"/L1_Summary.txt; do
    [[ -f "$seed_file" ]] || continue
    seed_destination="$OUTPUT_ROOT/$(basename "$seed_file")"
    if [[ ! -e "$seed_destination" ]]; then
      cp "$seed_file" "$seed_destination"
    fi
  done
fi

valid_level1_file() {
  local path="$1"
  local expected_buoy="$2"
  [[ -s "$path" ]] || return 1
  head -n 1 "$path" | grep -Eq \
    '^BuoyID,Year,Month,Day,Hour,(Min,Sec|Minute,Second),Lat,Lon,' || return 1
  awk -F, -v expected="$expected_buoy" '
    NR > 1 && $1 != expected {bad=1}
    END {exit (NR < 2 || bad)}
  ' "$path"
}

failure_count=0
target_count=0
validated_count=0
while IFS=, read -r buoy_id linked_images named_images ready_images track_holds platform_holds months blocks first_time last_time url destination; do
  [[ "$buoy_id" == "buoy_id" ]] && continue
  target_count=$((target_count + 1))
  expected_destination="$OUTPUT_ROOT/$buoy_id.csv"
  if [[ "$destination" != "$expected_destination" ]]; then
    echo "Unexpected destination for buoy $buoy_id: $destination" >&2
    exit 2
  fi
  if valid_level1_file "$destination" "$buoy_id"; then
    echo "SKIP verified IABP buoy $buoy_id"
    validated_count=$((validated_count + 1))
    continue
  fi

  partial_path="$destination.part"
  if valid_level1_file "$partial_path" "$buoy_id"; then
    mv "$partial_path" "$destination"
    echo "RECOVER verified IABP buoy $buoy_id"
    validated_count=$((validated_count + 1))
    continue
  fi
  if curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 30 \
    --max-time 180 \
    --retry 8 \
    --retry-delay 5 \
    --retry-all-errors \
    --output "$partial_path" \
    "$url" && valid_level1_file "$partial_path" "$buoy_id"; then
    mv "$partial_path" "$destination"
    echo "DONE IABP buoy $buoy_id"
    validated_count=$((validated_count + 1))
  else
    echo "FAILED IABP buoy $buoy_id" >&2
    failure_count=$((failure_count + 1))
  fi
done < "$TARGETS"

echo "IABP Level-1 target files present: $validated_count / $target_count"
if [[ "$failure_count" -gt 0 ]]; then
  exit 1
fi
