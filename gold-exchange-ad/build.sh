#!/usr/bin/env bash
# Full build of the Gold Exchange Lab 30 s ad.
#   ./build.sh <work_dir> [workers]
# <work_dir> must contain clips/, vo/, refs/logo.jpg (see README.md).
set -euo pipefail
WORK="$(cd "$1" && pwd)"
WORKERS="${2:-2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/scripts"
OUT="$WORK/out"
mkdir -p "$OUT"

echo "== audio"
python3 "$S/audio.py" "$WORK"
# two-pass loudness normalisation to -14 LUFS / -1.5 dBTP (social platforms)
M=$(ffmpeg -hide_banner -i "$OUT/mix_premaster.wav" -af aformat=channel_layouts=mono,loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
gv(){ echo "$M" | python3 -c "import sys,json; print(json.load(sys.stdin)['$1'])"; }
ffmpeg -v error -y -i "$OUT/mix_premaster.wav" -af "aformat=channel_layouts=mono,loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=$(gv input_i):measured_TP=$(gv input_tp):measured_LRA=$(gv input_lra):measured_thresh=$(gv input_thresh):offset=$(gv target_offset):linear=true,aresample=48000" -ac 2 -c:a pcm_s24le "$OUT/GOLD_EXCHANGE_LAB_MIX.wav"

echo "== subtitles"
python3 "$S/timeline.py" "$OUT/GOLD_EXCHANGE_LAB_SUBTITLES.srt" > /dev/null

echo "== picture"
python3 "$S/render.py" "$WORK" --workers "$WORKERS"

echo "== mux"
for kind in subs clean; do
  name=GOLD_EXCHANGE_LAB_FINAL_AD.mp4
  [ "$kind" = clean ] && name=GOLD_EXCHANGE_LAB_FINAL_AD_NO_SUBTITLES.mp4
  ffmpeg -v error -y -i "$OUT/_video_$kind.mp4" -i "$OUT/GOLD_EXCHANGE_LAB_MIX.wav" \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -ar 48000 -t 30 -movflags +faststart "$OUT/$name"
done

echo "== storyboard"
python3 "$S/storyboard.py" "$WORK"
rm -f "$OUT"/_seg_* "$OUT"/_*.txt
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate,nb_frames -of compact "$OUT/GOLD_EXCHANGE_LAB_FINAL_AD.mp4"
echo "build ok"
