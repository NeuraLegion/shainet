#!/bin/bash

# Build script for SHAInet CUDA kernels
# This will compile the CUDA kernels needed for GPU acceleration
# Called automatically during 'shards install' post-install

echo "Building SHAInet CUDA kernels..."

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc (CUDA compiler) not found."
    echo "CUDA kernels will not be built - GPU acceleration will be limited."
    echo "To enable full GPU acceleration:"
    echo "  1. Install CUDA toolkit"
    echo "  2. Run: ./build_cuda_kernels.sh"
    echo "  3. Set LD_LIBRARY_PATH to include this directory"
    exit 0  # Exit successfully to not break shards install
fi

# Set CUDA paths (adjust if needed)
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
SRC_DIR="src/shainet/native"
OUTPUT_LIB="libshainet_cuda_kernels.so"

# Compilation flags
NVCC_FLAGS="-shared --compiler-options=-fPIC -O3 -arch=sm_60"
INCLUDE_FLAGS="-I${CUDA_PATH}/include"
LIBRARY_FLAGS="-L${CUDA_PATH}/lib64 -lcurand"

# Check if source file exists
if [ ! -f "${SRC_DIR}/cuda_kernels.cu" ]; then
    echo "Warning: ${SRC_DIR}/cuda_kernels.cu not found"
    echo "CUDA kernels cannot be built - falling back to CPU-only mode"
    exit 0  # Exit successfully to not break shards install
fi

echo "Compiling CUDA kernels..."
nvcc ${NVCC_FLAGS} \
     ${INCLUDE_FLAGS} \
     ${LIBRARY_FLAGS} \
     -o ${OUTPUT_LIB} \
     ${SRC_DIR}/cuda_kernels.cu

if [ $? -eq 0 ]; then
    echo "✓ CUDA kernels compiled successfully: ${OUTPUT_LIB}"
    echo "✓ GPU acceleration is now available!"
    echo ""
    echo "Note: You may need to add this directory to LD_LIBRARY_PATH:"
    echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)"
    
    # Test if the library can be loaded
    if [ -f "${OUTPUT_LIB}" ]; then
        echo "✓ Library file created successfully"
        echo "  Size: $(du -h ${OUTPUT_LIB} | cut -f1)"
    fi
else
    echo "Warning: CUDA kernel compilation failed"
    echo "SHAInet will fall back to CPU-only mode"
    echo "To fix this issue:"
    echo "  1. Check CUDA installation"
    echo "  2. Verify CUDA toolkit version compatibility"
    echo "  3. Run: ./build_cuda_kernels.sh manually"
    exit 0  # Exit successfully to not break shards install
fi

echo ""
echo "SHAInet installation completed!"
echo ""
echo "GPU Acceleration Status:"
if [ -f "${OUTPUT_LIB}" ]; then
    echo "  ✓ CUDA kernels built successfully"
    echo "  ✓ GPU acceleration available"
    echo ""
    echo "To use GPU acceleration in your programs:"
    echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)"
    echo ""
    echo "Verify with: crystal eval \"require './src/shainet'; puts SHAInet::CUDA.kernels_available?\""
else
    echo "  ⚠ CUDA kernels not built - using CPU-only mode"
    echo "  ⚠ GPU acceleration limited to basic operations"
    echo ""
    echo "To enable full GPU acceleration later:"
    echo "  1. Install CUDA toolkit"
    echo "  2. Run: ./build_cuda_kernels.sh"
fi
