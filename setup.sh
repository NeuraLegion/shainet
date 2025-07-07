#!/bin/bash

# SHAInet Setup Script
# Installs dependencies and builds CUDA kernels for GPU acceleration

set -e

echo "🚀 SHAInet Setup"
echo "================"
echo ""

# Step 1: Install Crystal dependencies
echo "📦 Installing Crystal dependencies..."
if ! command -v shards &> /dev/null; then
    echo "❌ Error: 'shards' command not found."
    echo "Please install Crystal first: https://crystal-lang.org/install/"
    exit 1
fi

shards install
echo "✅ Crystal dependencies installed"
echo ""

# Step 2: Build CUDA kernels
echo "🔧 Building CUDA kernels for GPU acceleration..."
./build_cuda_kernels.sh
echo ""

# Step 3: Show final status and instructions
echo "🎉 Setup Complete!"
echo "=================="
echo ""

if [ -f "libshainet_cuda_kernels.so" ]; then
    echo "✅ CUDA kernels built successfully"
    echo "✅ GPU acceleration is ready!"
    echo ""
    echo "🔗 To use GPU acceleration, set the library path:"
    echo "   export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)"
    echo ""
    echo "💡 Add to your ~/.bashrc for permanent use:"
    echo "   echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)' >> ~/.bashrc"
    echo ""
    echo "🧪 Test GPU acceleration:"
    echo "   LD_LIBRARY_PATH=.:.\$LD_LIBRARY_PATH crystal eval \"require './src/shainet'; puts SHAInet::CUDA.kernels_available?\""
else
    echo "⚠️  CUDA kernels not built"
    echo "📊 GPU acceleration will be limited to basic operations"
    echo ""
    echo "💻 To enable full GPU acceleration:"
    echo "   1. Install CUDA toolkit from NVIDIA"
    echo "   2. Run: ./build_cuda_kernels.sh"
    echo "   3. Set LD_LIBRARY_PATH as shown above"
fi

echo ""
echo "📖 Next steps:"
echo "   - Check examples/ directory for usage examples"
echo "   - Run: crystal run examples/babylm_transformer.cr (optimized for GPU)"
echo "   - Monitor GPU usage: nvidia-smi -l 1"
echo ""
echo "🏃‍♂️ Quick start:"
echo "   make test    # Test GPU availability"
echo "   make help    # Show all available commands"
