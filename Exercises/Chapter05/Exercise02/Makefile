# 编译器和标志
NVCC = nvcc
CUDA_ARCH = sm_89
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
CXX_FLAGS = -std=c++17

TARGET = test_dynamic_tile
OBJECTS = solution.o test.o

all: $(TARGET)
	@echo "Build completed! Run 'make run'"

$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

solution.o: solution.cu
	$(NVCC) $(CUDA_FLAGS) -c $<

test.o: test.cpp
	$(NVCC) $(CXX_FLAGS) -c $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all run clean
