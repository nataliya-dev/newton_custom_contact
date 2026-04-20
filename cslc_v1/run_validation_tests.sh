#!/usr/bin/env bash
# CSLC regression validation.
#
# Runs the three coverage tests and flags any metric that drifts from the
# baseline recorded in cslc_v1/convo_april_20.md:
#   1. smooth_basic_test            — 33 unit tests must pass (§4.5)
#   2. squeeze_test  (mujoco)        — HoldDrop / HoldCreep per model (§1)
#   3. lift_test     (mujoco, fair)  — final_z, lifted, held per model (§2)
#
# Exit code: 0 if everything is within tolerance, 1 otherwise.
# Full logs are written to cslc_v1/_validation_logs/.

set -u
cd "$(dirname "$(readlink -f "$0")")/.."

LOG=cslc_v1/_validation_logs
mkdir -p "$LOG"
fails=0

banner() { echo; echo "=== $* ==="; }

# Tolerance ranges are intentionally loose enough to absorb run-to-run jitter
# (GPU nondeterminism, Warp JIT) but tight enough to catch a 2-3x regression.

# ---------------------------------------------------------------------------
# 1. smooth_basic_test
# ---------------------------------------------------------------------------
banner "smooth_basic_test (33 unit tests)"
f=$LOG/smooth_basic.log
if uv run --extra dev python -m unittest cslc_v1.smooth_basic_test -v >"$f" 2>&1; then
    if grep -Eq "^Ran 33 tests" "$f" && grep -q "^OK$" "$f"; then
        echo "  PASS -- 33/33"
    else
        echo "  FAIL -- unexpected test count or status; see $f"
        grep -E "^Ran|^FAILED|^OK" "$f" | sed 's/^/    /'
        fails=$((fails+1))
    fi
else
    echo "  FAIL -- test runner errored; see $f"
    fails=$((fails+1))
fi

# ---------------------------------------------------------------------------
# 2. squeeze_test                                      (ref: convo §1 table)
# ---------------------------------------------------------------------------
#   model           hold-drop [mm]    creep-rate [mm/s]
#   POINT_MUJOCO    +0.725            +0.495
#   CSLC_MUJOCO     +0.123            +0.082
#   HYDRO_MUJOCO    +0.093            +0.062
banner "squeeze_test (mujoco, point+cslc+hydro)"
f=$LOG/squeeze.log
if ! uv run cslc_v1/squeeze_test.py --mode squeeze --solver mujoco \
        --contact-models point,cslc,hydro >"$f" 2>&1; then
    echo "  FAIL -- squeeze_test crashed; see $f"
    fails=$((fails+1))
fi

# Pair each RESULT line with its most recent section header.
# _section prints "  MODEL_SOLVER" (two-space indent, caps + underscore).
awk '
/^  [A-Z_]+$/        { section=$1; next }
/RESULT:.*hold-drop/ { print section " " $0 }
' "$f" > "$LOG/squeeze.metrics"

check_squeeze() {
    # $1=model  $2=base_hold  $3=tol_hold  $4=base_creep  $5=tol_creep
    local model=$1 base_h=$2 tol_h=$3 base_c=$4 tol_c=$5
    local line; line=$(grep "^${model} " "$LOG/squeeze.metrics" || true)
    if [[ -z ${line} ]]; then
        echo "  FAIL ${model} -- no RESULT line in output"
        fails=$((fails+1)); return
    fi
    local hd cr
    hd=$(echo "$line" | sed -E 's/.*hold-drop=([+-]?[0-9.]+)mm.*/\1/')
    cr=$(echo "$line" | sed -E 's/.*creep-rate=([+-]?[0-9.]+)mm\/s.*/\1/')
    local ok=1
    awk -v v="$hd" -v b="$base_h" -v t="$tol_h" \
        'BEGIN{d=v-b; if(d<0)d=-d; exit !(d<=t)}' || ok=0
    awk -v v="$cr" -v b="$base_c" -v t="$tol_c" \
        'BEGIN{d=v-b; if(d<0)d=-d; exit !(d<=t)}' || ok=0
    if (( ok )); then
        printf "  PASS %-13s hold-drop=%+.3fmm  creep=%+.3fmm/s\n" \
            "$model" "$hd" "$cr"
    else
        printf "  FAIL %-13s hold-drop=%+.3f (base %+.3f+-%.2f)  creep=%+.3f (base %+.3f+-%.2f)\n" \
            "$model" "$hd" "$base_h" "$tol_h" "$cr" "$base_c" "$tol_c"
        fails=$((fails+1))
    fi
}

check_squeeze POINT_MUJOCO   0.725 0.30   0.495 0.20
check_squeeze CSLC_MUJOCO    0.123 0.30   0.082 0.20
check_squeeze HYDRO_MUJOCO   0.093 0.30   0.062 0.20

# ---------------------------------------------------------------------------
# 3. lift_test                                         (ref: convo §2 table)
# ---------------------------------------------------------------------------
#   model           final_z [m]
#   POINT_MUJOCO    0.0427
#   CSLC_MUJOCO     0.0494
#   HYDRO_MUJOCO    0.0466
# All three must report lifted=YES and held=YES; max_z must stay <= 0.053
# (hydro peaks at ~0.0517; point/cslc at ~0.0500).
banner "lift_test (mujoco, fair: --cslc-ka 15000 --kh 2.65e8)"
f=$LOG/lift.log
if ! uv run cslc_v1/lift_test.py --mode headless \
        --contact-models point,cslc,hydro \
        --cslc-ka 15000 --kh 2.65e8 >"$f" 2>&1; then
    echo "  FAIL -- lift_test crashed; see $f"
    fails=$((fails+1))
fi

awk '
/^  [A-Z_]+$/              { section=$1; next }
/RESULT:.*max_z=.*final_z/ { print section " " $0 }
' "$f" > "$LOG/lift.metrics"

check_lift() {
    # $1=model  $2=base_final_z  $3=tol_final_z
    local model=$1 base_fz=$2 tol_fz=$3
    local line; line=$(grep "^${model} " "$LOG/lift.metrics" || true)
    if [[ -z ${line} ]]; then
        echo "  FAIL ${model} -- no RESULT line in output"
        fails=$((fails+1)); return
    fi
    local mz fz lifted held
    mz=$(echo "$line"     | sed -E 's/.*max_z=([0-9.]+).*/\1/')
    fz=$(echo "$line"     | sed -E 's/.*final_z=([0-9.]+).*/\1/')
    lifted=$(echo "$line" | sed -E 's/.*lifted=([A-Z]+).*/\1/')
    held=$(echo "$line"   | sed -E 's/.*held=([A-Z]+).*/\1/')
    local ok=1
    awk -v v="$fz" -v b="$base_fz" -v t="$tol_fz" \
        'BEGIN{d=v-b; if(d<0)d=-d; exit !(d<=t)}' || ok=0
    [[ $lifted == YES ]] || ok=0
    [[ $held   == YES ]] || ok=0
    awk -v v="$mz" 'BEGIN{exit !(v <= 0.053)}' || ok=0
    if (( ok )); then
        printf "  PASS %-13s max_z=%.4f  final_z=%.4f  lifted=%s  held=%s\n" \
            "$model" "$mz" "$fz" "$lifted" "$held"
    else
        printf "  FAIL %-13s max_z=%.4f  final_z=%.4f (base %.4f+-%.3f)  lifted=%s  held=%s\n" \
            "$model" "$mz" "$fz" "$base_fz" "$tol_fz" "$lifted" "$held"
        fails=$((fails+1))
    fi
}

check_lift POINT_MUJOCO 0.0427 0.005
check_lift CSLC_MUJOCO  0.0494 0.005
check_lift HYDRO_MUJOCO 0.0466 0.005

# ---------------------------------------------------------------------------
banner "summary"
if (( fails == 0 )); then
    echo "  ALL CHECKS PASS"
    exit 0
else
    echo "  ${fails} CHECK(S) FAILED -- inspect logs in ${LOG}/"
    exit 1
fi
