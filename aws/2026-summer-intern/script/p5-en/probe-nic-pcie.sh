#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J probe-nic-pcie
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p5-en/%x-%j.out
#SBATCH -t 0:05:00
#SBATCH --nodelist=p5en-odcr-queue-dy-p5en48xlarge-33

set -e

echo "===================================================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "===================================================================="

echo ""
echo "==== 1) GPU: nvidia-smi PCIe link ===="
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
           --format=csv 2>&1 || echo "nvidia-smi failed"

echo ""
echo "==== 2) NIC: fi_info (EFA endpoints) ===="
if command -v fi_info >/dev/null 2>&1; then
    fi_info -p efa 2>&1 | head -80
else
    /opt/amazon/efa/bin/fi_info -p efa 2>&1 | head -80 || echo "fi_info not found"
fi

echo ""
echo "==== 3) NIC: /sys/class/infiniband link rate per port ===="
for p in /sys/class/infiniband/*/ports/*/rate; do
    [ -f "$p" ] && echo "$p: $(cat $p)"
done
for s in /sys/class/infiniband/*/ports/*/state; do
    [ -f "$s" ] && echo "$s: $(cat $s)"
done

echo ""
echo "==== 4) NIC: /sys/class/net link speed (EFA netdev interfaces) ===="
for d in /sys/class/net/*/speed; do
    dev=$(dirname "$d" | xargs basename)
    speed=$(cat "$d" 2>/dev/null || echo "n/a")
    echo "$dev: ${speed} Mbps"
done

echo ""
echo "==== 5) lspci: EFA / NIC devices (short) ===="
lspci | grep -iE "efa|elastic|amazon" || true

echo ""
echo "==== 6) lspci -vv: EFA PCIe LnkCap/LnkSta (needs no sudo for LnkCap; LnkSta may need sudo) ===="
# EFA vendor 1d0f device 1111 on AWS
for bdf in $(lspci -Dn | awk '$3 ~ /1d0f:1111/ {print $1}'); do
    echo "--- $bdf ---"
    lspci -s "$bdf" -vv 2>&1 | grep -E "^\s+(LnkCap|LnkSta|LnkCtl):" | head -4
done

echo ""
echo "==== 7) lspci -vv: NVIDIA GPU PCIe LnkCap/LnkSta ===="
for bdf in $(lspci -Dn | awk '$3 ~ /10de:/ {print $1}'); do
    # only H100/H200 (device id starts with 233x or 2321) - just include all NVIDIA for now
    echo "--- $bdf ---"
    lspci -s "$bdf" -vv 2>&1 | grep -E "^\s+(LnkCap|LnkSta|LnkCtl):" | head -4
done

echo ""
echo "==== 8) PCIe topology (nvidia-smi topo) ===="
nvidia-smi topo -m 2>&1 || true

echo ""
echo "==== 9) NUMA / CPU package count ===="
lscpu | grep -E "Socket|NUMA node|CPU\(s\):|Model name:" | head -8

echo ""
echo "==== done ===="
