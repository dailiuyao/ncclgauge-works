#!/bin/bash
# EFA Latency and Bandwidth Test Script
# Measures inter-node latency and bandwidth between two p6 instances using perftest

# Configuration
NODE1="p6-odcr-queue-dy-p6b20048xlarge-1"
NODE2="p6-odcr-queue-dy-p6b20048xlarge-2"
EFA_DEVICE="rdmap113s0"
CPU_CORE="48"  # CPU core pinned to NUMA node 1
PERFTEST_LAT_BIN="${HOME}/software/perftest/bin/ib_write_lat"
PERFTEST_BW_BIN="${HOME}/software/perftest/bin/ib_write_bw"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if perftest is built
if [ ! -f "${PERFTEST_LAT_BIN}" ] || [ ! -f "${PERFTEST_BW_BIN}" ]; then
    echo "ERROR: perftest not found"
    echo "  Latency binary: ${PERFTEST_LAT_BIN}"
    echo "  Bandwidth binary: ${PERFTEST_BW_BIN}"
    echo "Please run build_perftest.sh first to build perftest"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "==================================="
echo "EFA Latency and Bandwidth Test"
echo "==================================="
echo "Node 1 (Server): ${NODE1}"
echo "Node 2 (Client): ${NODE2}"
echo "EFA Device: ${EFA_DEVICE}"
echo "CPU Core: ${CPU_CORE}"
echo "Latency binary: ${PERFTEST_LAT_BIN}"
echo "Bandwidth binary: ${PERFTEST_BW_BIN}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "==================================="

# Check if we have an active Slurm allocation with both nodes
# Look for running jobs with 2 nodes on the correct partition
JOBID=$(squeue -u $USER -h -o "%i %N %t" -p p6-odcr-queue | grep " R$" | grep "\[1-2\]" | awk '{print $1}' | head -1)

if [ -z "$JOBID" ]; then
    echo "No active allocation found with both nodes. Allocating now..."
    salloc -N 2 -w "${NODE1},${NODE2}" -p p6-odcr-queue --no-shell &
    ALLOC_PID=$!
    echo "Waiting for allocation (PID: ${ALLOC_PID})..."
    sleep 15

    # Get job ID
    JOBID=$(squeue -u $USER -h -o "%i %N %t" -p p6-odcr-queue | grep " R$" | grep "\[1-2\]" | awk '{print $1}' | head -1)
    if [ -z "$JOBID" ]; then
        echo "ERROR: Failed to allocate nodes"
        echo "Current jobs:"
        squeue -u $USER
        exit 1
    fi
    echo "Allocated job ID: ${JOBID}"
else
    echo "Using existing job ID: ${JOBID}"
fi

# Verify nodes are ready
echo "Verifying node allocation..."
squeue -j ${JOBID}

echo ""
echo "==================================="
echo "Test 1: Latency Measurement"
echo "==================================="

# Start latency server on node 1 (in background)
echo "Starting latency server on ${NODE1}..."
srun -N 1 -w "${NODE1}" --jobid="${JOBID}" bash -c \
    "taskset -c ${CPU_CORE} ${PERFTEST_LAT_BIN} -c SRD -x 0 -F --report_gbits -Q 1 -t 1024 -a -d ${EFA_DEVICE} -n 10000 --inline_size 0 --write_with_imm" \
    > "${OUTPUT_DIR}/latency_server_${TIMESTAMP}.log" 2>&1 &

LAT_SERVER_PID=$!
echo "Latency server started (PID: ${LAT_SERVER_PID})"
sleep 5

# Run latency client on node 2
echo "Running latency client on ${NODE2}..."
srun -N 1 -w "${NODE2}" --jobid="${JOBID}" bash -c \
    "taskset -c ${CPU_CORE} ${PERFTEST_LAT_BIN} -c SRD -x 0 -F --report_gbits -Q 1 -t 1024 -a -d ${EFA_DEVICE} -n 10000 --inline_size 0 --write_with_imm ${NODE1}" \
    | tee "${OUTPUT_DIR}/latency_client_${TIMESTAMP}.log"

# Wait for latency server to finish
wait ${LAT_SERVER_PID}

echo "Latency test completed!"
echo ""
echo "==================================="
echo "Test 2: Bandwidth Measurement"
echo "==================================="

# Start bandwidth server on node 1 (in background)
echo "Starting bandwidth server on ${NODE1}..."
srun -N 1 -w "${NODE1}" --jobid="${JOBID}" bash -c \
    "taskset -c ${CPU_CORE} ${PERFTEST_BW_BIN} -c SRD -x 0 -F --report_gbits -Q 1 -a -d ${EFA_DEVICE} -n 5000 --inline_size 0 --write_with_imm" \
    > "${OUTPUT_DIR}/bandwidth_server_${TIMESTAMP}.log" 2>&1 &

BW_SERVER_PID=$!
echo "Bandwidth server started (PID: ${BW_SERVER_PID})"
sleep 5

# Run bandwidth client on node 2
echo "Running bandwidth client on ${NODE2}..."
srun -N 1 -w "${NODE2}" --jobid="${JOBID}" bash -c \
    "taskset -c ${CPU_CORE} ${PERFTEST_BW_BIN} -c SRD -x 0 -F --report_gbits -Q 1 -a -d ${EFA_DEVICE} -n 5000 --inline_size 0 --write_with_imm ${NODE1}" \
    | tee "${OUTPUT_DIR}/bandwidth_client_${TIMESTAMP}.log"

# Wait for bandwidth server to finish
wait ${BW_SERVER_PID}

echo "Bandwidth test completed!"
echo ""
echo "==================================="
echo "All Tests Completed!"
echo "==================================="
echo "Results saved to:"
echo "  Latency Server:   ${OUTPUT_DIR}/latency_server_${TIMESTAMP}.log"
echo "  Latency Client:   ${OUTPUT_DIR}/latency_client_${TIMESTAMP}.log"
echo "  Bandwidth Server: ${OUTPUT_DIR}/bandwidth_server_${TIMESTAMP}.log"
echo "  Bandwidth Client: ${OUTPUT_DIR}/bandwidth_client_${TIMESTAMP}.log"
echo "==================================="

# Create a summary file
cat > "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" << EOF
EFA Latency and Bandwidth Test Summary
=======================================
Date: $(date)
Node 1 (Server): ${NODE1}
Node 2 (Client): ${NODE2}
EFA Device: ${EFA_DEVICE}
CPU Core: ${CPU_CORE}

Latency Results (from client log):
===================================
EOF

# Extract key latency metrics from client log
grep -E "^ (2|64|1024|4096|65536|1048576|8388608) " "${OUTPUT_DIR}/latency_client_${TIMESTAMP}.log" >> "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" 2>/dev/null || true

cat >> "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" << EOF

Bandwidth Results (from client log):
=====================================
EOF

# Extract bandwidth metrics from client log
grep -E "^ (2|64|1024|4096|65536|1048576|8388608) " "${OUTPUT_DIR}/bandwidth_client_${TIMESTAMP}.log" >> "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" 2>/dev/null || true

echo ""
echo "Summary saved to: ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
