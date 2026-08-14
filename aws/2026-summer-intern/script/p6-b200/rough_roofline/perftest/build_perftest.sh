#!/bin/bash
# Build perftest for EFA latency testing
# This script builds perftest and installs it to /home/liuyaod/software/perftest

set -e  # Exit on error

# Configuration
PERFTEST_VERSION="25.10.0-0.128"
SOFTWARE_DIR="${HOME}/software"
PERFTEST_SRC="${SOFTWARE_DIR}/perftest-src"
PERFTEST_INSTALL="${SOFTWARE_DIR}/perftest"

echo "==================================="
echo "Building perftest ${PERFTEST_VERSION}"
echo "==================================="
echo "Source directory: ${PERFTEST_SRC}"
echo "Install directory: ${PERFTEST_INSTALL}"
echo "==================================="

# Create software directory if it doesn't exist
mkdir -p "${SOFTWARE_DIR}"

# Clean up existing source if it exists
if [ -d "${PERFTEST_SRC}" ]; then
    echo "Removing existing source directory..."
    rm -rf "${PERFTEST_SRC}"
fi

# Clone perftest repository
echo "Cloning perftest repository..."
cd "${SOFTWARE_DIR}"
git clone https://github.com/linux-rdma/perftest.git -b "${PERFTEST_VERSION}" perftest-src

# Build perftest
echo "Building perftest..."
cd "${PERFTEST_SRC}"

./autogen.sh

./configure \
  --prefix="${PERFTEST_INSTALL}" \
  CFLAGS=-I/opt/amazon/efa/include \
  CUDA_H_PATH=/usr/local/cuda/include

make -j $(nproc)
make install

echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo "Binaries installed to: ${PERFTEST_INSTALL}/bin/"
echo ""
echo "Available tools:"
ls -lh "${PERFTEST_INSTALL}/bin/" | grep -v "^total" | awk '{print "  - " $9}'
echo "==================================="
