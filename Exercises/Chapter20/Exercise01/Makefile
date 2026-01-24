# 第二十章 Exercise01 Makefile
# MPI + CUDA 分布式模板计算

NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60 -std=c++17

# MPI 编译标志（需要安装 OpenMPI 或 MPICH）
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile 2>/dev/null || echo "")
MPI_LINK_FLAGS = $(shell mpicc --showme:link 2>/dev/null || echo "")

# 检查 MPI 是否可用
ifeq ($(MPI_COMPILE_FLAGS),)
$(warning MPI not found. Please install OpenMPI or MPICH.)
$(warning On Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev)
$(warning On CentOS/RHEL: sudo yum install openmpi openmpi-devel)
endif

# 目标可执行文件
TARGET = stencil_mpi

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(CUDA_FLAGS) $(MPI_COMPILE_FLAGS) $(SOURCES) $(MPI_LINK_FLAGS) -o $(TARGET)

# 运行（3 个进程：2 计算节点 + 1 数据服务器）
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

# 运行（更多进程）
run4: $(TARGET)
	mpirun -np 4 ./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o

# Windows 清理
clean-win:
	del /F /Q $(TARGET).exe 2>nul || echo.

# 帮助
help:
	@echo "MPI + CUDA Stencil Computation"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the program"
	@echo "  run     - Run with 3 MPI processes"
	@echo "  run4    - Run with 4 MPI processes"
	@echo "  clean   - Remove build files"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit"
	@echo "  - MPI (OpenMPI or MPICH)"
	@echo ""
	@echo "Install MPI:"
	@echo "  Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev"
	@echo "  CentOS/RHEL:   sudo yum install openmpi openmpi-devel"

.PHONY: all run run4 clean clean-win help
