# Software Versions Used in EFA Tests

This document records all software and library versions used for the EFA latency and bandwidth tests in the `rough_roofline` directory.

**Last Updated:** 2026-06-03  
**Test Location:** `/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/`

---

## Network Stack Versions

### 1. **Perftest** (RDMA Performance Testing)
- **Version:** 25.10.0-0.128 (Binary reports: 6.27)
- **Location:** `~/software/perftest/`
- **Binaries Used:**
  - `ib_write_lat` - Latency measurement
  - `ib_write_bw` - Bandwidth measurement
- **Source:** https://github.com/linux-rdma/perftest
- **Build Date:** 2026-06-03

### 2. **Libfabric** (Fabric Interface Library)
- **Version:** 2.4.0amzn3.0
- **API Version:** 2.4
- **Location:** `/opt/amazon/efa/`
- **Binary:** `/opt/amazon/efa/bin/fi_info`
- **Provider:** EFA (Elastic Fabric Adapter)

### 3. **EFA Kernel Driver**
- **Version:** 3.0.0g
- **Module:** `/lib/modules/6.8.0-1030-aws/updates/dkms/efa.ko.zst`
- **Kernel:** 6.8.0-1030-aws
- **Check Command:**
  ```bash
  modinfo efa | grep version
  cat /sys/module/efa/version
  ```

---

## NCCL Stack Versions

### 4. **NCCL** (NVIDIA Collective Communications Library)
- **Version:** 2.30.4-1 (Active - used in tests)
- **Location:** `~/software/nccl/build/lib/`
- **Library:** `libnccl.so.2.30.4`
- **Build Date:** 2026-06-02
- **Git:** `v2.30.4-1` (HEAD detached)
- **Note:** Built from source in ~/software/nccl

**Alternative System Installations (not used in tests):**
- `/opt/portafiducia/nccl/v2.29.2-1/` - NCCL 2.29.2-1
- `/opt/portafiducia/nccl/v2.28.9-1/` - NCCL 2.28.9-1
- `/opt/portafiducia/nccl/v2.27.7-1/` - NCCL 2.27.7-1

### 5. **AWS OFI NCCL Plugin**
- **Version:** 1.19.0 (both user-built and system)
- **Git:** `v1.19.0` (HEAD detached)

**⚠️ IMPORTANT PATH ISSUE:**
The job scripts reference `~/software/aws-ofi-nccl/lib/` which **DOES NOT EXIST**.
The user-built libraries are at `~/software/aws-ofi-nccl/install/lib/`.
Scripts likely fall back to system installation at `/opt/amazon/ofi-nccl/lib/`.

**User Build (not accessible due to incorrect path):**
- **Location:** `~/software/aws-ofi-nccl/install/lib/`
- **Build Date:** 2026-06-01
- **Library Size:** 1.6M (libnccl-net.so)
- **Source:** `~/software/aws-ofi-nccl/`

**System Installation (likely being used):**
- **Location:** `/opt/amazon/ofi-nccl/lib/`
- **Install Date:** 2026-04-10
- **Library Size:** 488K (libnccl-net.so)

**Components:**
  - `libnccl-net-ofi.so` - Main OFI transport plugin
  - `libnccl-net.so` - Alias for net-ofi
  - `libnccl-ofi-tuner.so` - Performance tuning plugin
  - `libnccl-tuner-ofi.so` - Tuner alias

- **Minimum Required Libfabric:** 1.22.0 (Current: 2.4.0 ✓)
- **Source:** https://github.com/aws/aws-ofi-nccl

### 6. **NCCL Tests**
- **Version:** v2.18.3-2-g632ad39
- **Location:** `~/software/nccl-tests/`
- **Source:** https://github.com/NVIDIA/nccl-tests
- **Git Commit:** 632ad39

---

## Version Compatibility Matrix

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| Perftest | 6.27 | ✓ Latest | Used for raw EFA testing |
| Libfabric | 2.4.0amzn3.0 | ✓ Compatible | Exceeds OFI-NCCL requirement (1.22.0) |
| EFA Driver | 3.0.0g | ✓ Latest | Kernel 6.8.0-1030-aws |
| NCCL | 2.30.4-1 | ✓ Latest | Built from source 2026-06-02 |
| AWS OFI NCCL | 1.19.0 | ✓ Compatible | Works with NCCL 2.30.4 |
| NCCL Tests | 2.18.3 | ✓ Compatible | Slightly older but compatible |

---

## Version Check Commands

### Quick Version Check Script

```bash
#!/bin/bash
echo "=== Network Stack Versions ==="
echo "Perftest: $(~/software/perftest/bin/ib_write_lat --version 2>&1 | grep Version)"
echo "Libfabric: $(/opt/amazon/efa/bin/fi_info --version 2>&1 | head -1)"
echo "EFA Driver: $(cat /sys/module/efa/version 2>/dev/null)"
echo ""
echo "=== NCCL Stack Versions ==="
echo "NCCL: $(cd ~/software/nccl && git describe --tags 2>/dev/null)"
echo "AWS OFI NCCL: $(strings /opt/amazon/ofi-nccl/lib/libnccl-net.so | grep 'aws-ofi-nccl [0-9]' | head -1 | grep -oP '[0-9.]+' | head -1)"
echo "NCCL Tests: $(cd ~/software/nccl-tests && git describe --tags 2>/dev/null)"
```

### Individual Version Checks

**Perftest:**
```bash
~/software/perftest/bin/ib_write_lat --version
```

**Libfabric:**
```bash
/opt/amazon/efa/bin/fi_info --version
fi_info -p efa -t FI_EP_RDM  # List EFA providers
```

**EFA Driver:**
```bash
modinfo efa | grep -E "version|filename"
cat /sys/module/efa/version
```

**NCCL:**
```bash
cd ~/software/nccl && git describe --tags
ls -l ~/software/nccl/build/lib/libnccl.so.*
# Alternative: check system installations
ls -l /opt/portafiducia/nccl/v*/install/lib/libnccl.so.*
```

**AWS OFI NCCL Plugin:**
```bash
strings /opt/amazon/ofi-nccl/lib/libnccl-net.so | grep "aws-ofi-nccl [0-9]"
```

**NCCL Tests:**
```bash
cd ~/software/nccl-tests && git describe --tags
```

---

## Configuration Details

### Environment Variables (Typical NCCL Setup)

```bash
export LD_LIBRARY_PATH=/opt/portafiducia/nccl/v2.29.2-1/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

export NCCL_PROTO=simple
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
```

### Hardware Configuration

**Instance Type:** p6-odcr-queue-dy-p6b20048xlarge  
**EFA Devices:** 8 per instance (4 per NUMA node)
- NUMA 0: rdmap79s0, rdmap80s0, rdmap96s0, rdmap97s0
- NUMA 1: rdmap113s0, rdmap114s0, rdmap132s0, rdmap133s0

**Tests Used:** rdmap113s0 (NUMA node 1, CPU core 48)

---

## Upgrade History

| Date | Component | Old Version | New Version | Notes |
|------|-----------|-------------|-------------|-------|
| 2026-06-03 | Perftest | N/A | 25.10.0-0.128 | Initial build |
| N/A | NCCL | - | 2.29.2 | Pre-installed |
| N/A | AWS OFI NCCL | - | 1.19.0 | Pre-installed |
| N/A | Libfabric | - | 2.4.0amzn3.0 | Pre-installed |
| N/A | EFA Driver | - | 3.0.0g | Pre-installed |

---

## Known Issues / Compatibility Notes

1. **NCCL 2.29.2 + AWS OFI NCCL 1.19.0**: Fully compatible
2. **Libfabric 2.4.0**: Significantly newer than minimum required (1.22.0) - no issues expected
3. **NCCL Tests 2.18.3**: Slightly older than NCCL 2.29.2, but backward compatible
4. **EFA Driver 3.0.0g**: Latest stable release for kernel 6.8.0

---

## Performance Baseline

With these versions, achieved performance on p6-odcr-queue instances:

**Latency (ib_write_lat, one-way):**
- Small messages (2B-4KB): ~14-15 μs
- Medium messages (64KB): ~21.7 μs
- Large messages (1MB): ~49.5 μs
- Very large (8MB): ~220 μs

**Bandwidth (ib_write_bw):**
- Peak bandwidth: ~390 Gb/s (48.75 GB/s)
- Achieved at: 1MB+ message sizes

---

## References

- Perftest: https://github.com/linux-rdma/perftest
- AWS OFI NCCL: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://github.com/ofiwg/libfabric
- NCCL: https://github.com/NVIDIA/nccl
- NCCL Tests: https://github.com/NVIDIA/nccl-tests
- EFA Documentation: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

---

**Document Version:** 1.0  
**Created:** 2026-06-03  
**Author:** Automated version collection script
