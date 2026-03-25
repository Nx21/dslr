# Matrix Library Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Python / venv settings
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
PYTHON_STAMP = .python_built

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

# Default target: build everything
all: $(TARGET) python

# ── Static library ────────────────────────────────────────────────
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(TARGET): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $^
	@echo "Library $(LIB_NAME) created successfully"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# ── Virtual environment ───────────────────────────────────────────
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Virtual environment created"

venv: $(VENV)/bin/activate

# ── Python bindings ───────────────────────────────────────────────
$(PYTHON_STAMP): $(VENV)/bin/activate setup.py $(SRC_DIR)/python_bindings.cpp $(HEADERS)
	$(PIP) install pybind11
	$(PIP) install -e .
	touch $(PYTHON_STAMP)
	@echo "Python bindings built successfully"

python: $(PYTHON_STAMP)

# ── Run a Python script ───────────────────────────────────────────
# Usage: make run SCRIPT=your_script.py
run: python
	$(PYTHON) $(SCRIPT)

# ── Debug build ───────────────────────────────────────────────────
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

# ── Example ───────────────────────────────────────────────────────
example: $(TARGET) example.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -L$(LIB_DIR) example.cpp -lmatrix -o example

# ── Clean ─────────────────────────────────────────────────────────
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) build *.egg-info $(PYTHON_STAMP)
	rm -f example
	@echo "Cleaned build artifacts"

clean-all: clean
	rm -rf $(VENV)
	@echo "Cleaned everything including virtual environment"

# ── Rebuild ───────────────────────────────────────────────────────
rebuild: clean all

# ── Install system-wide ───────────────────────────────────────────
install: $(TARGET)
	sudo cp -r $(INCLUDE_DIR)/* /usr/local/include/
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "Matrix library installed to system"

uninstall:
	sudo rm -f /usr/local/lib/$(LIB_NAME)
	sudo rm -rf /usr/local/include/Matrix*
	@echo "Matrix library uninstalled from system"

# ── Help ──────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  all        - Build static library + Python bindings (default)"
	@echo "  python     - Build Python bindings only"
	@echo "  run        - Run a Python script: make run SCRIPT=your_script.py"
	@echo "  debug      - Build with debug flags"
	@echo "  example    - Build example program"
	@echo "  clean      - Remove build artifacts"
	@echo "  clean-all  - Remove build artifacts + virtual environment"
	@echo "  rebuild    - Clean and build"
	@echo "  install    - Install library system-wide"
	@echo "  uninstall  - Remove library from system"
	@echo "  help       - Show this help message"

.PHONY: all python venv run debug example clean clean-all rebuild install uninstall help