# Stats Library Makefile

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
PYTHON = $(VENV)/bin/python3.14
PIP = $(VENV)/bin/pip
PYTHON_STAMP = .python_built

# Target library name
LIB_NAME = libstats.a
TARGET = $(LIB_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/descriptive.cpp \
          $(SRC_DIR)/quartiles.cpp \
          $(SRC_DIR)/correlation.cpp \
          $(SRC_DIR)/normalization.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Header files
HEADERS = $(INCLUDE_DIR)/Stats.hpp

# Default target: build everything
all: $(TARGET) python

# -- Static library ------------------------------------------------
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(TARGET): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $^
	@echo "Library $(LIB_NAME) created successfully"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# -- Virtual environment ------------------------------------------
$(VENV)/bin/activate:
	python3.14 -m venv $(VENV)
	@echo "Virtual environment created"

venv: $(VENV)/bin/activate

# -- Python bindings ----------------------------------------------
$(PYTHON_STAMP): $(VENV)/bin/activate setup.py $(SRC_DIR)/python_bindings.cpp $(HEADERS)
	$(PIP) install pybind11
	$(PIP) install -e .
	touch $(PYTHON_STAMP)
	@echo "Python bindings built successfully"

python: $(PYTHON_STAMP)

# -- Run a Python script ------------------------------------------
# Usage: make run SCRIPT=your_script.py
run: python
	$(PYTHON) $(SCRIPT)

# -- Example -------------------------------------------------------
example: python
	$(MAKE) run SCRIPT=example.py

# -- Debug build ---------------------------------------------------
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

# -- Clean ---------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) build *.egg-info $(PYTHON_STAMP)
	rm -f stats*.so
	@echo "Cleaned build artifacts"

clean-all: clean
	rm -rf $(VENV)
	@echo "Cleaned everything including virtual environment"

# -- Rebuild -------------------------------------------------------
rebuild: clean all

# -- Install system-wide ------------------------------------------
install: $(TARGET)
	sudo cp -r $(INCLUDE_DIR)/* /usr/local/include/
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "Stats library installed to system"

uninstall:
	sudo rm -f /usr/local/lib/$(LIB_NAME)
	sudo rm -f /usr/local/include/Stats.hpp
	@echo "Stats library uninstalled from system"

# -- Help ----------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  all        - Build static library + Python bindings (default)"
	@echo "  python     - Build Python bindings only"
	@echo "  run        - Run a Python script: make run SCRIPT=your_script.py"
	@echo "  example    - Run example.py"
	@echo "  debug      - Build with debug flags"
	@echo "  clean      - Remove build artifacts"
	@echo "  clean-all  - Remove build artifacts + virtual environment"
	@echo "  rebuild    - Clean and build"
	@echo "  install    - Install library system-wide"
	@echo "  uninstall  - Remove library from system"
	@echo "  help       - Show this help message"

.PHONY: all python venv run example debug clean clean-all rebuild install uninstall help

