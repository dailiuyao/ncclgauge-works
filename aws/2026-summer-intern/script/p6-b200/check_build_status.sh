#!/bin/bash

echo "=========================================="
echo "P6 B200 Build Status Check"
echo "=========================================="
echo ""

echo "1. NCCL Library Status:"
if [ -f "/home/liuyaod/software/nccl_profile/build/lib/libnccl.so" ]; then
    echo "   ✓ NCCL library built"
    ls -lh /home/liuyaod/software/nccl_profile/build/lib/libnccl.so*
else
    echo "   ✗ NCCL library not found"
fi

echo ""
echo "2. AWS OFI Plugin Status:"
if [ -f "/home/liuyaod/software/aws-ofi-nccl_profile/install/lib/libnccl-net.so" ]; then
    echo "   ✓ AWS OFI plugin built"
    ls -lh /home/liuyaod/software/aws-ofi-nccl_profile/install/lib/libnccl-net.so*
else
    echo "   ✗ AWS OFI plugin not found"
fi

echo ""
echo "3. Gauge Binaries Status (P6 B200):"
GAUGE_DIR="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/gauge/p6-b200"
if [ -d "$GAUGE_DIR" ]; then
    if [ -f "$GAUGE_DIR/pping_gauge_n_1.exe" ]; then
        echo "   ✓ Gauge binary built"
        ls -lh $GAUGE_DIR/*.exe
    else
        echo "   ⚠ Gauge directory exists but binaries missing:"
        ls -lh $GAUGE_DIR/ 2>/dev/null || echo "   Directory empty"
    fi
else
    echo "   ✗ Gauge p6-b200 directory not created yet"
fi

echo ""
echo "4. Output Directory Status:"
OUT_DIR="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p6-b200"
if [ -d "$OUT_DIR" ]; then
    echo "   ✓ Output directory exists"
    file_count=$(ls -1 $OUT_DIR/*.out 2>/dev/null | wc -l)
    if [ $file_count -gt 0 ]; then
        echo "   Found $file_count output files:"
        ls -lth $OUT_DIR/*.out | head -5
    else
        echo "   No output files yet"
    fi
else
    echo "   ✗ Output directory not created"
fi

echo ""
echo "=========================================="
echo "Build Actions Needed:"
echo "=========================================="

NEEDS_NCCL=false
NEEDS_PLUGIN=false
NEEDS_GAUGE=false

if [ ! -f "/home/liuyaod/software/nccl_profile/build/lib/libnccl.so" ]; then
    echo "[ ] Run: bash nccl_profile_build.sh"
    NEEDS_NCCL=true
fi

if [ ! -f "/home/liuyaod/software/aws-ofi-nccl_profile/install/lib/libnccl-net.so" ]; then
    echo "[ ] Run: bash nccl_profile_build.sh (includes plugin)"
    NEEDS_PLUGIN=true
fi

if [ ! -f "$GAUGE_DIR/pping_gauge_n_1.exe" ]; then
    echo "[ ] Run: bash pping-gauge-build.sh"
    NEEDS_GAUGE=true
fi

if [ "$NEEDS_NCCL" = false ] && [ "$NEEDS_PLUGIN" = false ] && [ "$NEEDS_GAUGE" = false ]; then
    echo "✓ All components built! Ready to run tests."
    echo ""
    echo "Next step: sbatch pping-gauge-run-vary-msg.sh"
fi

echo ""
