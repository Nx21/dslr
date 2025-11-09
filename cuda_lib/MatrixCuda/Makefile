# CUDA Matrix Library Makefile

# Compiler settings
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++17 -O2 -Xcompiler -Wall,-Wextra
DEBUG_FLAGS = -g -G -DDEBUG
CUDA_ARCH = -arch=sm_89  # RTX 4070 compute capability
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# CUDA installation path (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Target library name
LIB_NAME = libmatrixcuda.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(wildcard $(INCLUDE_DIR)/*.cuh)

# CUDA libraries
CUDA_LIBS = -lcudart -lcurand

# Default target
all: $(TARGET)

# Create directories if they don't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

# Build static library
$(TARGET): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $^
	@echo "CUDA Library $(LIB_NAME) created successfully"

# Compile CUDA object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -I$(INCLUDE_DIR) -I$(CUDA_INC) -dc $< -o $@

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Test target (if you want to add tests later)
test: $(TARGET)
	@echo "Running CUDA tests..."
	# Add test commands here when you have tests

# Example program to test the CUDA library
example: $(TARGET) example.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -I$(INCLUDE_DIR) -L$(LIB_DIR) example.cu -lmatrixcuda $(CUDA_LIBS) -o example

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
	rm -f example
	@echo "Cleaned build artifacts"

# Force rebuild
rebuild: clean all

# Install (copy headers and library to system directories)
install: $(TARGET)
	sudo cp -r $(INCLUDE_DIR)/* /usr/local/include/
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "CUDA Matrix library installed to system"

# Uninstall
uninstall:
	sudo rm -f /usr/local/lib/$(LIB_NAME)
	sudo rm -rf /usr/local/include/MatrixCuda*
	@echo "CUDA Matrix library uninstalled from system"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc || echo "NVCC not found in PATH"
	@nvcc --version 2>/dev/null || echo "NVCC not working properly"
	@ls $(CUDA_PATH) 2>/dev/null || echo "CUDA path $(CUDA_PATH) not found"

# Check GPU info
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
	@echo "\nCUDA Device Info:"
	@echo '__global__ void dummy() {} int main() { int count; cudaGetDeviceCount(&count); printf("CUDA Devices: %d\\n", count); return 0; }' | $(NVCC) -x cu - -o /tmp/cuda_test && /tmp/cuda_test && rm -f /tmp/cuda_test || echo "Cannot detect CUDA devices"

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the CUDA library (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  test       - Run tests"
	@echo "  example    - Build example program"
	@echo "  clean      - Remove build artifacts"
	@echo "  rebuild    - Clean and build"
	@echo "  install    - Install library system-wide"
	@echo "  uninstall  - Remove library from system"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  gpu-info   - Show GPU and CUDA device information"
	@echo "  help       - Show this help message"

# Declare phony targets
.PHONY: all debug test example clean rebuild install uninstall check-cuda gpu-info help

# Show build information
info:
	@echo "NVCC: $(NVCC)"
	@echo "Flags: $(NVCC_FLAGS)"
	@echo "Architecture: $(CUDA_ARCH)"
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Target: $(TARGET)"