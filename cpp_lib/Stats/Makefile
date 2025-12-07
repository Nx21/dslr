# Stats Library Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Target library name
LIB_NAME = libstats.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/stats.cpp \
          $(SRC_DIR)/descriptive.cpp \
          $(SRC_DIR)/quartiles.cpp \
          $(SRC_DIR)/correlation.cpp \
          $(SRC_DIR)/normalization.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(INCLUDE_DIR)/Stats.hpp

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
	@echo "Library $(LIB_NAME) created successfully"

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
	@echo "Cleaned build artifacts"

# Phony targets
.PHONY: all debug clean

