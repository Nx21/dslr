# CUDA Stats Library Makefile

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
CUDA_PATH ?= /usr/local/cuda-12.0
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Target library name
LIB_NAME = libstatscuda.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/stats.cu \
          $(SRC_DIR)/kernels.cu \
          $(SRC_DIR)/gpu_operations.cu

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(INCLUDE_DIR)/StatsCuda.cuh

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

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
	@echo "Cleaned build artifacts"

# Phony targets
.PHONY: all debug clean

