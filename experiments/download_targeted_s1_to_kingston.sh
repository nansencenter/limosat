#!/usr/bin/env bash
set -euo pipefail

URL_FILE="${URL_FILE:-results/iabp_s1_stratified_coverage/tier1_asf_urls.txt}"
DEST_ROOT="${DEST_ROOT:-/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020/sentinel1/raw}"
AUTH_NETRC="${AUTH_NETRC:-/Users/seachu/.netrc}"
PARALLEL="${PARALLEL:-4}"
MAX_FILES="${MAX_FILES:-0}"

case "$DEST_ROOT" in
  /Volumes/KINGSTON/*) ;;
  *)
    echo "Refusing non-KINGSTON destination: $DEST_ROOT" >&2
    exit 2
    ;;
esac

if [[ ! -f "$URL_FILE" ]]; then
  echo "Missing URL file: $URL_FILE" >&2
  exit 2
fi
if [[ ! -f "$AUTH_NETRC" ]]; then
  echo "Missing Earthdata netrc: $AUTH_NETRC" >&2
  exit 2
fi

download_one() {
  local url="$1"
  local name date year month out_dir out part bytes cookie_file
  local -a resume_args=()
  name="$(basename "${url%%\?*}")"
  if [[ "$name" =~ _([0-9]{8})T[0-9]{6}_ ]]; then
    date="${BASH_REMATCH[1]}"
    year="${date:0:4}"
    month="${date:4:2}"
  else
    echo "Cannot parse acquisition month: $name" >&2
    return 2
  fi
  out_dir="$DEST_ROOT/$year/$month"
  out="$out_dir/$name"
  part="$out.part"
  mkdir -p "$out_dir"
  if [[ -s "$out" ]]; then
    if unzip -tq "$out" >/dev/null; then
      echo "SKIP verified $name"
      return 0
    fi
    echo "Existing ZIP failed verification: $out" >&2
    return 3
  fi
  echo "START $name"
  if ! cookie_file="$(mktemp "${TMPDIR:-/tmp}/limosat_asf_cookie.XXXXXX")"; then
    echo "FAILED creating temporary cookie jar for $name" >&2
    return 8
  fi
  chmod 600 "$cookie_file"
  trap 'rm -f "$cookie_file"' EXIT
  if [[ -s "$part" ]]; then
    resume_args=(-C -)
  fi
  if ! curl -L --fail --silent --show-error --retry 8 --retry-delay 5 \
    --retry-all-errors --connect-timeout 30 --max-time 0 \
    "${resume_args[@]}" --netrc-file "$AUTH_NETRC" \
    --cookie "$cookie_file" --cookie-jar "$cookie_file" \
    -o "$part" "$url"; then
    echo "FAILED download $name" >&2
    return 4
  fi
  if ! unzip -tq "$part" >/dev/null; then
    echo "FAILED ZIP verification $name" >&2
    return 5
  fi
  if ! mv -f "$part" "$out"; then
    echo "FAILED finalizing $name" >&2
    return 6
  fi
  if ! bytes="$(stat -f %z "$out")"; then
    echo "FAILED size check $name" >&2
    return 7
  fi
  echo "DONE $name $bytes bytes"
}

export -f download_one
export DEST_ROOT AUTH_NETRC

if [[ "$MAX_FILES" -gt 0 ]]; then
  sed -n "1,${MAX_FILES}p" "$URL_FILE" |
    xargs -P "$PARALLEL" -n 1 bash -c 'download_one "$1"' _
else
  sed '/^[[:space:]]*$/d' "$URL_FILE" |
    xargs -P "$PARALLEL" -n 1 bash -c 'download_one "$1"' _
fi
