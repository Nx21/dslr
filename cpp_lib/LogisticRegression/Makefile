# Linear Regression Library Makefile

CXX = g++
CXX_FLAGS = -std=c++17 -O2 -Wall -Wextra
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Target library name
LIB_NAME = liblinearregression.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/construction.cpp \
		  $(SRC_DIR)/training.cpp \
		  $(SRC_DIR)/prediction.cpp \
		  $(SRC_DIR)/uri.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(INCLUDE_DIR)/LinearRegression.hpp

# Paths to dependencies
MATRIX_INC = -I../Matrix/include
STATS_INC = -I../Stats/include

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
	@echo "Linear Regression Library $(LIB_NAME) created successfully"

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) $(MATRIX_INC) $(STATS_INC) -I$(INCLUDE_DIR) -c $< -o $@

# Debug build
debug: CXX_FLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
	@echo "Cleaned build artifacts"

# Force rebuild
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all    - Build the library (default)"
	@echo "  debug  - Build with debug flags"
	@echo "  clean  - Remove build artifacts"
	@echo "  rebuild - Clean and build"
	@echo "  help   - Show this help message"

# Declare phony targets
.PHONY: all debug clean rebuild help

# Show build information
info:
	@echo "CXX: $(CXX)"
	@echo "Flags: $(CXX_FLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Target: $(TARGET)"
