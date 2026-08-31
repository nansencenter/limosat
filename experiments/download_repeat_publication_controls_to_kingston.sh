#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_MANIFEST="$ROOT_DIR/results/iabp_s1_stratified_coverage/full70_repeat_publication_controls.csv"

if [[ ! -d /Volumes/KINGSTON ]]; then
  echo "KINGSTON is not mounted." >&2
  exit 2
fi
if [[ ! -f "$CONTROL_MANIFEST" ]]; then
  echo "Repeat-publication control manifest is missing: $CONTROL_MANIFEST" >&2
  exit 2
fi

failure_count=0
while IFS=, read -r control_id image_time primary_id primary_name primary_zip primary_vae repeat_id repeat_name url destination repeat_vae overlap purpose; do
  [[ "$control_id" == "repeat_control_id" ]] && continue
  if [[ "$destination" != /Volumes/KINGSTON/* ]]; then
    echo "Refusing non-KINGSTON destination: $destination" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]] && unzip -tq "$destination" >/dev/null; then
    echo "SKIP verified repeat publication $repeat_name"
    continue
  fi
  partial_path="$destination.part"
  if curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 30 \
    --max-time 900 \
    --retry 8 \
    --retry-delay 10 \
    --retry-all-errors \
    --output "$partial_path" \
    "$url" && unzip -tq "$partial_path" >/dev/null; then
    mv "$partial_path" "$destination"
    echo "DONE repeat publication $repeat_name"
  else
    echo "FAILED repeat publication $repeat_name" >&2
    failure_count=$((failure_count + 1))
  fi
done < "$CONTROL_MANIFEST"

if [[ "$failure_count" -gt 0 ]]; then
  exit 1
fi
