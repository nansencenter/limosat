#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-experiments/configs/multisensor_march_expansion_20260819.json}"
AUTH_NETRC="${AUTH_NETRC:-/Users/seachu/.netrc}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing frozen manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$AUTH_NETRC" ]]; then
  echo "Missing Earthdata netrc: $AUTH_NETRC" >&2
  exit 2
fi
if [[ "$(jq -r '.selection_status' "$MANIFEST")" != "frozen_before_download_or_deformation_association" ]]; then
  echo "Manifest is not marked frozen: $MANIFEST" >&2
  exit 2
fi

while IFS=$'\t' read -r url output; do
  case "$output" in
    /Volumes/KINGSTON/*) ;;
    *)
      echo "Refusing non-KINGSTON output: $output" >&2
      exit 2
      ;;
  esac
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]] && h5dump -H "$output" >/dev/null 2>&1; then
    echo "SKIP verified $(basename "$output")"
    continue
  fi
  part="${output}.part"
  cookie_file="$(mktemp "${TMPDIR:-/tmp}/limosat_icesat_cookie.XXXXXX")"
  chmod 600 "$cookie_file"
  resume_offset=0
  if [[ -s "$part" ]]; then
    resume_offset=-
  fi
  echo "START $(basename "$output")"
  if ! curl -L --fail --silent --show-error --retry 8 --retry-delay 5 \
    --retry-all-errors --connect-timeout 30 --max-time 0 \
    -C "$resume_offset" --netrc-file "$AUTH_NETRC" \
    --cookie "$cookie_file" --cookie-jar "$cookie_file" \
    -o "$part" "$url"; then
    rm -f "$cookie_file"
    echo "FAILED $(basename "$output")" >&2
    exit 4
  fi
  rm -f "$cookie_file"
  if ! h5dump -H "$part" >/dev/null 2>&1; then
    echo "FAILED HDF5 verification: $part" >&2
    exit 5
  fi
  mv -f "$part" "$output"
  shasum -a 256 "$output" > "${output}.sha256"
  echo "DONE $(basename "$output") $(stat -f %z "$output") bytes"
done < <(jq -r '.candidates[] | [.download_url, .output_path] | @tsv' "$MANIFEST")
