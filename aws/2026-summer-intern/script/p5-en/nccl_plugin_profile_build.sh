#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J nccl-plugin-profile-build
# SLURM does not expand env vars in #SBATCH directives; output goes to the
# sbatch submit directory. cd to the desired log dir before submitting.
#SBATCH -o %x-%j.out
#SBATCH -t 0:30:00
# Build on a FABRIC_1.9 node so the plugin links against the newer libfabric
# symver. Known-old (FABRIC_1.8-only) nodes: 15-22, 40, 41, 42, 44, 45, 46.
# The matching run script must exclude the same set, since a FABRIC_1.9-linked
# libnccl-net.so cannot dlopen on a FABRIC_1.8-only host.
#SBATCH --exclude=p5en-odcr-queue-dy-p5en48xlarge-[15-22,40-42,44-46]

set -e

# ============================================================================
# User-configurable paths — MUST be exported in the shell before `sbatch`.
# sbatch copies this script to /var/spool/slurmd/... at exec time, so
# BASH_SOURCE cannot resolve the repo — you must set these explicitly.
#   NCCL_GAUGE_HOME  : path to this repo's root (contains gauge/, script/, out/)
#   NCCL_SRC         : path to the NCCL profile source tree
#   AWS_OFI_NCCL_SRC : path to the aws-ofi-nccl profile source tree
# ============================================================================
export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"
export NCCL_SRC="/home/liuyaod/software/nccl_profile"
export AWS_OFI_NCCL_SRC="/home/liuyaod/software/aws-ofi-nccl_profile"

: "${NCCL_GAUGE_HOME:?please export NCCL_GAUGE_HOME to the root of this gauge tree}"
: "${NCCL_SRC:?please export NCCL_SRC to the nccl_profile source tree}"
: "${AWS_OFI_NCCL_SRC:?please export AWS_OFI_NCCL_SRC to the aws-ofi-nccl_profile source tree}"

# ---- Fixed toolchain locations (edit here if your cluster differs) --------
# Pin CUDA to 12.8: p5en nodes' default /usr/local/cuda points to 13.1 which
# exceeds the driver's supported CUDA version (13.0 per nvidia-smi), causing
# cuGetProcAddress PFN lookups to fail and NCCL to SEGV during init.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export MPI_HOME=${MPI_HOME:-/opt/amazon/openmpi}
export LIBFABRIC_HOME=${LIBFABRIC_HOME:-/opt/amazon/efa}
# P5 uses Hopper architecture (compute capability 9.0)
export NVCC_GENCODE=${NVCC_GENCODE:-"-gencode=arch=compute_90,code=sm_90"}

# ---- Derived paths --------------------------------------------------------
export NCCL_HOME=${NCCL_HOME:-${NCCL_SRC}/build}
# Directory holding libprofiling_arrays.so (defines the gauge↔plugin shared
# timing arrays). The plugin links against this lib and bakes its directory
# into RPATH so it resolves at runtime.
export PROFILING_LIB_DIR=${PROFILING_LIB_DIR:-${NCCL_GAUGE_HOME}/gauge}

export LD_LIBRARY_PATH=${MPI_HOME}/lib:${LD_LIBRARY_PATH}

# Redirect nvcc/gcc temp files off the root partition. NCCL device-code
# compilation writes tens of GB of tmpxft_*.cpp*.ii and cc*.s into $TMPDIR;
# the default /tmp lives on / which is ~94% full on this host.
export TMPDIR=${TMPDIR:-/home/liuyaod/tmp}
mkdir -p "${TMPDIR}"

echo "=== Build host ==="
echo "  hostname           = $(hostname)"
echo "  libfabric versions available (should include what runtime nodes have):"
objdump -T "${LIBFABRIC_HOME}/lib/libfabric.so.1" 2>/dev/null \
    | grep -oE 'FABRIC_[0-9.]+' | sort -u | sed 's/^/    /'
echo ""

echo "=== Build configuration ==="
echo "  CUDA_HOME          = ${CUDA_HOME}"
echo "  MPI_HOME           = ${MPI_HOME}"
echo "  LIBFABRIC_HOME     = ${LIBFABRIC_HOME}"
echo "  NCCL_SRC           = ${NCCL_SRC}"
echo "  AWS_OFI_NCCL_SRC   = ${AWS_OFI_NCCL_SRC}"
echo "  NCCL_GAUGE_HOME    = ${NCCL_GAUGE_HOME}"
echo "  PROFILING_LIB_DIR  = ${PROFILING_LIB_DIR}"
echo "  NVCC_GENCODE       = ${NVCC_GENCODE}"
echo "  TMPDIR             = ${TMPDIR}"
echo ""

# ---- Build libprofiling_arrays.so ----------------------------------------
# Owns the gauge↔plugin shared timing arrays (profiling_arrays.cpp). Must be
# rebuilt whenever profiling_interface.h or profiling_arrays.cpp changes,
# otherwise the plugin's `-Wl,--allow-shlib-undefined` link will succeed but
# runtime dlopen will fail on any newly-added symbol.
echo "=== Building libprofiling_arrays.so ==="
pushd "${PROFILING_LIB_DIR}"
if [ ! -f profiling_arrays.cpp ] || [ ! -f profiling_interface.h ]; then
    echo "ERROR: profiling_arrays.cpp / profiling_interface.h not found in ${PROFILING_LIB_DIR}"
    exit 1
fi
g++ -shared -fPIC -O2 -std=c++17 -o libprofiling_arrays.so profiling_arrays.cpp
popd

# ---- Build NCCL (profile version) -----------------------------------------
# Using v2.27.6 approach: build library only (skip utilities like ncclparam).
# The library has undefined symbols for profiling variables (gauge_*) which
# are defined in the gauge executable, not in NCCL.
echo "=== Building NCCL (profile) for P5 ==="
pushd "${NCCL_SRC}/src"
# Force a clean rebuild: prior build linked against a different CUDA toolkit
# (13.1 vs the driver's 13.0 support), producing a binary whose PFN queries
# fail on this driver.
make -C .. clean || true
make -j$(nproc) lib NVCC_GENCODE="${NVCC_GENCODE}"
popd

# ---- Build aws-ofi-nccl (profile version) ---------------------------------
# Pass the PROFILING_LIB_DIR via LDFLAGS so the Makefile.am's plain
# `-lprofiling_arrays` resolves and gets baked into RPATH for runtime.
echo "=== Building aws-ofi-nccl (profile) ==="
pushd "${AWS_OFI_NCCL_SRC}"
# Force full recompile of every .o/.lo. Objects from a prior build on another
# host will bind fi_* symbols to that host's libfabric symver (e.g. FABRIC_1.9),
# and the linker on this host would silently reuse them.
if [ -f Makefile ]; then
    make distclean || true
fi
./autogen.sh
# CPPFLAGS: let the patched source find profiling_interface.h (it lives in                                                             
# PROFILING_LIB_DIR alongside profiling_arrays.cpp), so the include can stay                                                           
# relative (#include "profiling_interface.h") instead of a hardcoded path.                                                             
CPPFLAGS="-I${PROFILING_LIB_DIR}" \
LDFLAGS="-L${PROFILING_LIB_DIR} -Wl,-rpath,${PROFILING_LIB_DIR}" \
./configure --prefix=${AWS_OFI_NCCL_SRC}/install \
            --with-mpi=${MPI_HOME} \
            --with-libfabric=${LIBFABRIC_HOME} \
            --with-nccl=${NCCL_HOME} \
            --with-cuda=${CUDA_HOME}
make -C src -j$(nproc) && make -C src install
popd

echo "=== Verifying plugin's libfabric dependency ==="
objdump -T "${AWS_OFI_NCCL_SRC}/install/lib/libnccl-net.so" 2>/dev/null \
    | grep -oE 'FABRIC_[0-9.]+' | sort -u | sed 's/^/  /'

echo "=== Profile build complete for P5 ==="
