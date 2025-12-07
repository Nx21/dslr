# Matrix Library Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Target library name
LIB_NAME = libmatrix.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/constructors.cpp \
          $(SRC_DIR)/functions.cpp \
          $(SRC_DIR)/operations.cpp \
          $(SRC_DIR)/property.cpp \
          $(SRC_DIR)/vectorOperations.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(INCLUDE_DIR)/Matrix.hpp

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

# Test target (if you want to add tests later)
test: $(TARGET)
	@echo "Running tests..."
	# Add test commands here when you have tests

# Example program to test the library
example: $(TARGET) example.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -L$(LIB_DIR) example.cpp -lmatrix -o example

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
	@echo "Matrix library installed to system"

# Uninstall
uninstall:
	sudo rm -f /usr/local/lib/$(LIB_NAME)
	sudo rm -rf /usr/local/include/Matrix*
	@echo "Matrix library uninstalled from system"

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build the library (default)"
	@echo "  debug    - Build with debug flags"
	@echo "  test     - Run tests"
	@echo "  example  - Build example program"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and build"
	@echo "  install  - Install library system-wide"
	@echo "  uninstall- Remove library from system"
	@echo "  help     - Show this help message"

# Declare phony targets
.PHONY: all debug test example clean rebuild install uninstall help

# Show build information
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Target: $(TARGET)"