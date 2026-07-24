#!/bin/bash
# Wait for the audit agents to finish (load settles), then run the decisive test on a
# free machine. Avoids the thundering-herd that SIGTERM'd the previous attempt.
cd /home/LTTakahashi/NullState/probe-demo
for i in $(seq 1 180); do   # up to ~60 min of waiting
  L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  if [ "$L" -lt 12 ]; then echo "[wait] load=$L after $((i*20))s -> starting decisive test"; break; fi
  sleep 20
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec python3 -u run_decisive.py
