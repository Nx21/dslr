# Main Project Makefile
# Orchestrates building all libraries and main executable

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
NVCC_FLAGS = -std=c++17 -O2 -Xcompiler -Wall,-Wextra
CUDA_ARCH = -arch=sm_89

# Project directories
CPP_LIB_DIR = cpp_lib
CUDA_LIB_DIR = cuda_lib
OBJ_DIR = obj

# Library paths
MATRIX_DIR = $(CPP_LIB_DIR)/Matrix
STATS_DIR = $(CPP_LIB_DIR)/Stats
MATRIX_CUDA_DIR = $(CUDA_LIB_DIR)/MatrixCuda
STATS_CUDA_DIR = $(CUDA_LIB_DIR)/Stats

# Library files
LIBMATRIX = $(MATRIX_DIR)/lib/libmatrix.a
LIBSTATS = $(STATS_DIR)/lib/libstats.a
LIBMATRIX_CUDA = $(MATRIX_CUDA_DIR)/lib/libmatrixcuda.a
LIBSTATS_CUDA = $(STATS_CUDA_DIR)/lib/libstatscuda.a

# Include paths
INCLUDES = -I$(MATRIX_DIR)/include \
           -I$(STATS_DIR)/include \
           -I$(MATRIX_CUDA_DIR)/include \
           -I$(STATS_CUDA_DIR)/include

# CUDA paths
CUDA_PATH ?= /usr/local/cuda-12.0
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64
CUDA_LIBS = -lcudart -lcurand

# Default target - build all libraries
.PHONY: all
all: cpp-libs cuda-libs
	@echo "All libraries built successfully"

# Build C++ libraries
.PHONY: cpp-libs
cpp-libs: matrix stats
	@echo "C++ libraries built successfully"

# Build CUDA libraries
.PHONY: cuda-libs
cuda-libs: matrix-cuda stats-cuda
	@echo "CUDA libraries built successfully"

# Build individual libraries
.PHONY: matrix
matrix:
	@echo "Building Matrix library..."
	$(MAKE) -C $(MATRIX_DIR)

.PHONY: stats
stats:
	@echo "Building Stats library..."
	$(MAKE) -C $(STATS_DIR)

.PHONY: matrix-cuda
matrix-cuda:
	@echo "Building Matrix CUDA library..."
	$(MAKE) -C $(MATRIX_CUDA_DIR)

.PHONY: stats-cuda
stats-cuda:
	@echo "Building Stats CUDA library..."
	$(MAKE) -C $(STATS_CUDA_DIR)

# Build main executable (C++ version)
.PHONY: main
main: cpp-libs main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		main.cpp \
		-L$(MATRIX_DIR)/lib -lmatrix \
		-L$(STATS_DIR)/lib -lstats \
		-o main
	@echo "Main executable built successfully"

# Build main executable (CUDA version)
.PHONY: main-cuda
main-cuda: cuda-libs main_cuda.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -I$(CUDA_INC) \
		main_cuda.cpp \
		-L$(MATRIX_CUDA_DIR)/lib -lmatrixcuda \
		-L$(STATS_CUDA_DIR)/lib -lstatscuda \
		-L$(CUDA_LIB) $(CUDA_LIBS) \
		-o main-cuda
	@echo "Main CUDA executable built successfully"

# Build both versions
.PHONY: main-all
main-all: main main-cuda
	@echo "All executables built successfully"

# Create object directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean all libraries
.PHONY: clean
clean:
	@echo "Cleaning all libraries..."
	$(MAKE) -C $(MATRIX_DIR) clean
	$(MAKE) -C $(STATS_DIR) clean
	$(MAKE) -C $(MATRIX_CUDA_DIR) clean
	$(MAKE) -C $(STATS_CUDA_DIR) clean
	rm -rf $(OBJ_DIR)
	rm -f main main-cuda
	@echo "Cleaned all build artifacts"

# Clean only C++ libraries
.PHONY: clean-cpp
clean-cpp:
	@echo "Cleaning C++ libraries..."
	$(MAKE) -C $(MATRIX_DIR) clean
	$(MAKE) -C $(STATS_DIR) clean

# Clean only CUDA libraries
.PHONY: clean-cuda
clean-cuda:
	@echo "Cleaning CUDA libraries..."
	$(MAKE) -C $(MATRIX_CUDA_DIR) clean
	$(MAKE) -C $(STATS_CUDA_DIR) clean

# Rebuild everything
.PHONY: rebuild
rebuild: clean all

# Show library status
.PHONY: status
status:
	@echo "=== Library Status ==="
	@echo "C++ Libraries:"
	@echo -n "  Matrix:      "; test -f $(LIBMATRIX) && echo "✓ Built" || echo "✗ Not built"
	@echo -n "  Stats:       "; test -f $(LIBSTATS) && echo "✓ Built" || echo "✗ Not built"
	@echo "CUDA Libraries:"
	@echo -n "  Matrix CUDA: "; test -f $(LIBMATRIX_CUDA) && echo "✓ Built" || echo "✗ Not built"
	@echo -n "  Stats CUDA:  "; test -f $(LIBSTATS_CUDA) && echo "✓ Built" || echo "✗ Not built"
	@echo "Executables:"
	@echo -n "  main:        "; test -f main && echo "✓ Built" || echo "✗ Not built"
	@echo -n "  main-cuda:   "; test -f main-cuda && echo "✓ Built" || echo "✗ Not built"

# Help message
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Build all libraries (default)"
	@echo "  cpp-libs     - Build only C++ libraries (Matrix + Stats)"
	@echo "  cuda-libs    - Build only CUDA libraries (MatrixCuda + StatsCuda)"
	@echo "  matrix       - Build Matrix library"
	@echo "  stats        - Build Stats library"
	@echo "  matrix-cuda  - Build Matrix CUDA library"
	@echo "  stats-cuda   - Build Stats CUDA library"
	@echo "  main         - Build main executable (C++ version)"
	@echo "  main-cuda    - Build main executable (CUDA version)"
	@echo "  main-all     - Build both main executables"
	@echo "  clean        - Clean all libraries and executables"
	@echo "  clean-cpp    - Clean only C++ libraries"
	@echo "  clean-cuda   - Clean only CUDA libraries"
	@echo "  rebuild      - Clean and rebuild all"
	@echo "  status       - Show build status of all libraries"
	@echo "  help         - Show this help message"

