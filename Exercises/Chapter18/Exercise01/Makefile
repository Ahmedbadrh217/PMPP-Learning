# 第十八章 Exercise01 Makefile
# 静电势能图计算

NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60 -std=c++17
INCLUDES = -I../../../Common

# 目标可执行文件
TARGET = cenergy_benchmark

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o

# Windows 清理
clean-win:
	del /F /Q $(TARGET).exe 2>nul || echo.

.PHONY: all run clean clean-win
