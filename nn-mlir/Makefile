# MLIR NN Compiler Makefile

# Configuration
LLVM_PREFIX := /opt/homebrew/Cellar/llvm@17/17.0.6
LLVM_BIN := $(LLVM_PREFIX)/bin
LLVM_INC := $(LLVM_PREFIX)/include
LLVM_LIB := $(LLVM_PREFIX)/lib

# Tools
TBLGEN := $(LLVM_BIN)/mlir-tblgen
CXX := clang++

# Flags
CXXFLAGS := -std=c++17 -fno-rtti -fno-exceptions -I$(LLVM_INC) -Iinclude -Ibuild/include
LDFLAGS := -L$(LLVM_LIB) -lMLIR -lLLVM
TBLGEN_FLAGS := -I$(LLVM_INC) -Iinclude

# Directories
GEN_DIR := build/include/NN

.PHONY: all clean tablegen

all: tablegen nn-opt nn-compiler

tablegen: $(GEN_DIR)/NNDialect.h.inc $(GEN_DIR)/NNDialect.cpp.inc $(GEN_DIR)/NNOps.h.inc $(GEN_DIR)/NNOps.cpp.inc

$(GEN_DIR)/NNDialect.h.inc: include/NN/NNOps.td
	mkdir -p $(GEN_DIR)
	$(TBLGEN) --gen-dialect-decls --dialect=nn $(TBLGEN_FLAGS) $< -o $@

$(GEN_DIR)/NNDialect.cpp.inc: include/NN/NNOps.td
	mkdir -p $(GEN_DIR)
	$(TBLGEN) --gen-dialect-defs --dialect=nn $(TBLGEN_FLAGS) $< -o $@

$(GEN_DIR)/NNOps.h.inc: include/NN/NNOps.td
	mkdir -p $(GEN_DIR)
	$(TBLGEN) --gen-op-decls $(TBLGEN_FLAGS) $< -o $@

$(GEN_DIR)/NNOps.cpp.inc: include/NN/NNOps.td
	mkdir -p $(GEN_DIR)
	$(TBLGEN) --gen-op-defs $(TBLGEN_FLAGS) $< -o $@

build/NNDialect.o: lib/NN/NNDialect.cpp include/NN/NNDialect.h $(GEN_DIR)/NNDialect.h.inc $(GEN_DIR)/NNDialect.cpp.inc
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/NNOps.o: lib/NN/NNOps.cpp include/NN/NNOps.h $(GEN_DIR)/NNOps.h.inc $(GEN_DIR)/NNOps.cpp.inc
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

nn-opt: tablegen tools/nn-opt.cpp build/NNDialect.o build/NNOps.o
	$(CXX) $(CXXFLAGS) tools/nn-opt.cpp build/NNDialect.o build/NNOps.o $(LDFLAGS) -o $@

nn-compiler: tablegen tools/nn-compiler.cpp build/NNDialect.o build/NNOps.o
	$(CXX) $(CXXFLAGS) tools/nn-compiler.cpp build/NNDialect.o build/NNOps.o $(LDFLAGS) -o $@

clean:
	rm -rf build nn-opt nn-compiler
