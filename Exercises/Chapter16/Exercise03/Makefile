# ============================================================================
# Makefile - 第十六章: CNN 完整层实现
# ============================================================================

# 编译器和标志
NVCC = nvcc

# CUDA 架构
CUDA_ARCH = sm_89
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
CXX_FLAGS = -std=c++17

# Common 目录路径
COMMON_DIR = ../../../Common

# 目标可执行文件
TARGET = test_cnn_layers

# 源文件
CUDA_SRC = solution.cu
CPP_SRC = test.cpp

# 目标文件
OBJECTS = solution.o test.o

# 默认目标
all: $(TARGET)
	@echo "Build completed successfully!"
	@echo "Run 'make run' to execute the program"

# 链接目标文件
$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

# 编译 CUDA 源文件
solution.o: $(CUDA_SRC) solution.h
	$(NVCC) $(CUDA_FLAGS) -I$(COMMON_DIR) -c $<

# 编译 C++ 测试文件
test.o: $(CPP_SRC) solution.h
	$(NVCC) $(CXX_FLAGS) -I$(COMMON_DIR) -c $<

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Cleaned all build files"

help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build artifacts"

.PHONY: all run clean help
