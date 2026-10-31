#!/usr/bin/env bash
# Prepare personal clips: keep the first MAX seconds, strip metadata, re-encode.
# Usage: bash scripts/prepare_clips.sh SRC_DIR DEST_DIR [MAX_SECONDS] [MIN_SECONDS]
set -euo pipefail

src="${1:?source folder}"
dest="${2:?destination folder}"
max="${3:-20}"
min="${4:-10}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found"; exit 1; }
[ -d "$src" ] || { echo "source folder not found: $src"; exit 1; }
mkdir -p "$dest"
map="$dest/_map.csv"
if [ -e "$map" ]; then
  echo "refusing to overwrite $map; move or rename the destination folder first"
  exit 1
fi
echo "clip,original_name,original_seconds" > "$map"

n=0
skipped=0
failed=0
while IFS= read -r -d '' file; do
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$file" 2>/dev/null | head -n1 || true)
  if ! awk -v d="$dur" -v m="$min" 'BEGIN { exit !(d + 0 >= m) }'; then
    skipped=$((skipped + 1))
    continue
  fi
  idx=$((n + 1))
  printf -v name "p_%03d.mp4" "$idx"
  out="$dest/$name"
  if ffmpeg -nostdin -v error -y -i "$file" -t "$max" \
      -map 0:v:0 -map "0:a:0?" \
      -map_metadata -1 -map_metadata:s:v -1 -map_metadata:s:a -1 -map_chapters -1 \
      -fflags +bitexact -flags:v +bitexact -flags:a +bitexact \
      -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
      -c:a aac -b:a 128k -movflags +faststart "$out"; then
    n=$idx
    printf '%s,"%s",%s\n' "$name" "${file//\"/\"\"}" "$dur" >> "$map"
  else
    rm -f "$out"
    failed=$((failed + 1))
  fi
done < <(find "$src" -maxdepth 1 -type f \( -iname '*.mov' -o -iname '*.mp4' -o -iname '*.m4v' \) -print0 | sort -z)

leftovers=0
for f in "$dest"/p_*.mp4; do
  [ -e "$f" ] || continue
  if ffprobe -v error -show_format -show_streams -of default=nw=1 "$f" 2>/dev/null \
      | grep -qiE '^TAG:.*(location|creation_time|com\.apple|gps|iso6709)'; then
    leftovers=$((leftovers + 1))
  fi
done

echo "processed: $n"
echo "skipped (shorter than $min s): $skipped"
echo "failed: $failed"
echo "files with leftover location/date tags: $leftovers"
