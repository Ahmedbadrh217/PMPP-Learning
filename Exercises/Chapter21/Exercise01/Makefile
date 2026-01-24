# 第二十一章 Exercise01 Makefile
# Bezier 曲线自适应细分（动态并行）

# 编译器设置
NVCC = nvcc

# GPU 架构（动态并行需要 sm_35+）
CUDA_ARCH = sm_60

# 动态并行编译选项
# -rdc=true: 启用可重定位设备代码
# -lcudadevrt: 链接设备运行时库
NVCC_FLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -rdc=true
LIBS = -lcudadevrt

# 目标
TARGET = bezier_test

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) $(LIBS) -o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean
