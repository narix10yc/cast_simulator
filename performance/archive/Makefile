# Compiler and flags
CLANG = $(llvm_root)/bin/clang
CLANGXX = $(llvm_root)/bin/clang++
# CLANG = /opt/homebrew/opt/llvm/bin/clang
# CLANGXX = /opt/homebrew/opt/llvm/bin/clang++

BUILD_DIR = $(CURDIR)/../build
INCLUDE_DIR = $(CURDIR)/../include

# MACOS_SDK_PATH = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
# CFLAGS = -std=c++17 -I$(INCLUDE_DIR) -Ofast -march=native -isysroot $(MACOS_SDK_PATH)
IR_CFLAGS = -march=native -O1 -Wno-override-module
CFLAGS = -std=c++17 -I$(INCLUDE_DIR) -O3 -march=native
LDFLAGS = -L$(BUILD_DIR)/timeit -ltimeit -lpthread

# Directories
IR_DIR = $(CURDIR)/gen_file
OBJ_DIR = $(CURDIR)/obj
SRC_FILE = $(CURDIR)/benchmark.cpp
HEADER_FILE =$(wildcard $(CURDIR)/*.h)

# Files
IR_FILES = $(wildcard $(IR_DIR)/*.ll)
OBJ_FILES = $(patsubst $(IR_DIR)/%.ll, $(OBJ_DIR)/%.ll.o, $(IR_FILES))

# Default target
all: $(OBJ_DIR) benchmark single_gate

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile LLVM IR files to object files
$(OBJ_DIR)/%.ll.o: $(IR_DIR)/%.ll | $(OBJ_DIR)
	$(CLANG) $(IR_CFLAGS) -c -o $@ $<

fidelity: $(CURDIR)/gen_file.ll $(CURDIR)/fidelity.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/fidelity
	
benchmark: $(CURDIR)/gen_file.ll $(CURDIR)/benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/benchmark

benchmark_m: $(OBJ_FILES) $(CURDIR)/benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/benchmark

gate_benchmark : $(CURDIR)/gen_file.ll $(CURDIR)/gate_benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/gate_benchmark

full_gate_bcmk : $(CURDIR)/gen_file.ll $(CURDIR)/full_gate_bcmk.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/full_gate_bcmk

single_gate : $(CURDIR)/gen_file.ll $(CURDIR)/single_gate.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/single_gate

correctness: $(CURDIR)/gen_file.ll $(CURDIR)/correctness.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -L$(BUILD_DIR)/cast -lcast -o $(BUILD_DIR)/correctness

ir_benchmark : ir_benchmark_gcc ir_benchmark_clang

ir_benchmark_gcc: $(OBJ_FILES) $(CURDIR)/ir_benchmark.cpp
	g++ $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/ir_benchmark_gcc

ir_benchmark_clang: $(OBJ_FILES) $(CURDIR)/ir_benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/ir_benchmark_clang

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o

# Phony targets
.PHONY: all clean benchmark ir_benchmark
