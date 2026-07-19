#!/usr/bin/env bash
# Recover summer_s2 members past the corrupt 2MB gap (see emrdm_reevaluation.md
# §2.1/§2.2). Assembles prefix + zero-gap + downloaded tail, runs gzrecover,
# and extracts the summer-73 clear patches. Recovery is NOT trusted here — the
# validation gate (verify_summer73.py) runs afterward.
#
# Requires: gzrecover (apt package `gzrt`). Verify with `command -v gzrecover`.
#
# Usage: recover_summer_s2.sh <data_root>
set -euo pipefail
ROOT="${1:-$HOME/data/sen12mscr}"
ARC_DIR="$ROOT/_archives"
PREFIX="$ARC_DIR/ROIs1868_summer_s2.tar.gz"   # readable prefix [0, OFF)
TAIL="$ARC_DIR/summer_s2_tail.bin"            # downloaded [OFF+2MB, END)
ASSEMBLED="$ARC_DIR/summer_s2_recovered.tar.gz"
OFF=16609967160
HOLE=2097152
ARCHIVE_END=40141465440

command -v gzrecover >/dev/null || { echo "FAIL: gzrecover not installed (apt install gzrt)"; exit 3; }

prefix_sz=$(stat -c %s "$PREFIX")
tail_sz=$(stat -c %s "$TAIL")
[ "$prefix_sz" -eq "$OFF" ] || { echo "FAIL: prefix is $prefix_sz B, expected $OFF"; exit 1; }
want_tail=$((ARCHIVE_END - OFF - HOLE))
[ "$tail_sz" -eq "$want_tail" ] || { echo "FAIL: tail is $tail_sz B, expected $want_tail"; exit 1; }

echo "assembling: prefix($prefix_sz) + zeros($HOLE) + tail($tail_sz)"
cp --reflink=auto "$PREFIX" "$ASSEMBLED"
head -c "$HOLE" /dev/zero >> "$ASSEMBLED"
cat "$TAIL" >> "$ASSEMBLED"
asm_sz=$(stat -c %s "$ASSEMBLED")
[ "$asm_sz" -eq "$ARCHIVE_END" ] || { echo "FAIL: assembled $asm_sz != $ARCHIVE_END"; exit 1; }
echo "assembled full-length archive: $asm_sz B"

# gzrecover writes <name>.recovered (a raw tar stream with the corrupt region
# skipped). Then extract only the summer-73 clear members.
echo "running gzrecover (this reads all 40GB; slow)..."
gzrecover -o "$ARC_DIR/summer_s2_recovered.tar" "$ASSEMBLED"
echo "extracting ROIs1868_summer_s2/s2_73/* from recovered tar..."
tar -xf "$ARC_DIR/summer_s2_recovered.tar" --ignore-zeros --warning=no-unknown-keyword \
  -C "$ROOT" 'ROIs1868_summer_s2/s2_73/*' 2>/dev/null || true

n73=$(ls "$ROOT/ROIs1868_summer_s2/s2_73/"*.tif 2>/dev/null | wc -l)
echo "RECOVERED scene 73 clear patches: $n73 (reference 783)"
# Clean the large intermediates; keep prefix+tail for re-runs.
rm -f "$ASSEMBLED" "$ARC_DIR/summer_s2_recovered.tar"
echo "RECOVERY_DONE n73=$n73"
