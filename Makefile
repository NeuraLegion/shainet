# Makefile for SHAInet CUDA optimization
.PHONY: cuda clean test help

# CUDA paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
NVCC ?= nvcc

# Build targets
CUDA_LIB = libshainet_cuda_kernels.so
CUDA_SRC = src/shainet/native/cuda_kernels.cu

# Compilation flags
NVCC_FLAGS = -shared --compiler-options=-fPIC -O3 -arch=sm_60 -Wno-deprecated-gpu-targets
INCLUDE_FLAGS = -I$(CUDA_PATH)/include
LIBRARY_FLAGS = -L$(CUDA_PATH)/lib64 -lcurand

help:
	@echo "SHAInet CUDA Optimization Makefile"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install  - Install dependencies and build CUDA kernels (recommended)"
	@echo "  cuda     - Build CUDA kernels for GPU acceleration"
	@echo "  test     - Test if CUDA kernels are working"
	@echo "  clean    - Remove built libraries"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make install                 # Complete setup (dependencies + CUDA)"
	@echo "  make cuda                    # Build CUDA kernels only"
	@echo "  LD_LIBRARY_PATH=. make test  # Test with library path"
	@echo ""
	@echo "For GPU acceleration, use 'make install' instead of 'shards install'"

cuda: $(CUDA_LIB)

$(CUDA_LIB): $(CUDA_SRC)
	@echo "Building CUDA kernels..."
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "Error: $(NVCC) not found. Please install CUDA toolkit."; \
		exit 1; \
	fi
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(LIBRARY_FLAGS) -o $@ $<
	@echo "✓ CUDA kernels built successfully: $@"
	@echo "✓ Library size: $$(du -h $@ | cut -f1)"
	@echo ""
	@echo "To use GPU acceleration, set the library path:"
	@echo "  export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$$(pwd)"

install:
	@echo "Installing SHAInet dependencies and building CUDA kernels..."
	@echo "Step 1: Installing Crystal dependencies..."
	shards install
	@echo "Step 2: Building CUDA kernels for GPU acceleration..."
	@./build_cuda_kernels.sh
	@echo ""
	@echo "Installation complete!"
	@echo ""
	@if [ -f "libshainet_cuda_kernels.so" ]; then \
		echo "✓ CUDA kernels built successfully"; \
		echo "✓ GPU acceleration is available"; \
		echo ""; \
		echo "To use GPU acceleration in your programs:"; \
		echo "  export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$$(pwd)"; \
		echo ""; \
		echo "Add this to your ~/.bashrc for persistent use:"; \
		echo "  echo 'export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$$(pwd)' >> ~/.bashrc"; \
	else \
		echo "⚠ CUDA kernels not built - GPU acceleration limited"; \
		echo "⚠ Install CUDA toolkit and run 'make cuda' to enable full GPU support"; \
	fi

test:
	@echo "Testing CUDA kernel availability..."
	@crystal eval "require \"./src/shainet\"; \
		puts \"CUDA available: #{SHAInet::CUDA.available?}\"; \
		puts \"CUDA kernels available: #{SHAInet::CUDA.kernels_available?}\"; \
		if SHAInet::CUDA.kernels_available?; \
			puts \"✓ GPU acceleration is ready!\"; \
		else; \
			puts \"✗ GPU acceleration not available. Run 'make cuda' first.\"; \
		end"

clean:
	rm -f $(CUDA_LIB)
	@echo "✓ Cleaned built libraries"

# GPU benchmark target (optional)
benchmark: $(CUDA_LIB)
	@echo "Running GPU vs CPU benchmark..."
	@LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH crystal run benchmarks/matrix_benchmark.cr

.DEFAULT_GOAL := help
