#!/usr/bin/env bash
# Archive the shared pair-product store to NIRD, or restore it from there.
#   sync_pair_store_nird.sh archive   /cluster/work store -> NIRD (after a run's pairs are done)
#   sync_pair_store_nird.sh restore   NIRD -> /cluster/work store (e.g. after work-area cleanup)
# Run it on the NIRD transfer node (ssh svc01 from Olivia): Olivia's login and compute
# nodes do not mount NIRD. Products are immutable, so files already present are never
# overwritten; data files are copied before their markers, so a marker never points at
# a partly copied data file. Override WORK_STORE / NIRD_STORE to use other locations.
set -euo pipefail

direction="${1:-}"
work_store="${WORK_STORE:-/cluster/work/projects/nn9878k/seachu/eloftr-pair-store}"
nird_store="${NIRD_STORE:-/nird/datapeak/NS2993K/seachu/eloftr-pair-store}"
case "$direction" in
  archive) source="$work_store"; destination="$nird_store" ;;
  restore) source="$nird_store"; destination="$work_store" ;;
  *) echo "usage: $0 archive|restore" >&2; exit 2 ;;
esac
[[ -d "$source" ]] || { echo "missing source store: $source" >&2; exit 2; }
mkdir -p "$destination"
common=(--recursive --ignore-existing --times --prune-empty-dirs --exclude '.*.writing.*')
rsync "${common[@]}" --include '*/' --include '*.npz' --exclude '*' "$source/" "$destination/"
rsync "${common[@]}" --include '*/' --include '*.json' --exclude '*' "$source/" "$destination/"
echo "synced $(find "$destination" -name '*.json' | wc -l) markers in $destination"
