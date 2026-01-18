/**
 * 第十三章：排序 - 测试程序
 * 
 * 参考：chapter-13/code/main.cu
 * 
 * 测试5种排序实现的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "solution.h"

// 验证数组是否有序
bool isSorted(unsigned int* arr, int N) {
    for (int i = 0; i < N - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("  错误：索引 %d 处无序 (%u > %u)\n", i, arr[i], arr[i + 1]);
            return false;
        }
    }
    return true;
}

// 复制数组到设备并排序
void testSort(const char* name, void (*sortFunc)(unsigned int*, int), 
              const unsigned int* h_original, int N) {
    printf("%s...\n", name);
    
    // 分配设备内存
    unsigned int* d_arr;
    cudaMalloc(&d_arr, N * sizeof(unsigned int));
    cudaMemcpy(d_arr, h_original, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // 排序
    sortFunc(d_arr, N);
    
    // 拷贝回主机验证
    unsigned int* h_sorted = new unsigned int[N];
    cudaMemcpy(h_sorted, d_arr, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    if (isSorted(h_sorted, N)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果错误！\n\n");
    }
    
    delete[] h_sorted;
    cudaFree(d_arr);
}

// 包装函数用于多位基数排序
void gpuRadixSortMultibitWrapper(unsigned int* d_input, int N) {
    gpuRadixSortMultibit(d_input, N, RADIX_BITS);
}

void gpuRadixSortCoarsenedWrapper(unsigned int* d_input, int N) {
    gpuRadixSortCoarsened(d_input, N, RADIX_BITS);
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十三章：排序\n");
    printf("  Parallel Sorting - Multiple Implementations\n");
    printf("  参考: chapter-13/code/\n");
    printf("================================================================\n\n");
    
    // 测试参数
    const int N = 1000000;  // 100万元素
    
    printf("配置:\n");
    printf("  数组长度: %d\n", N);
    printf("  BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("  RADIX_BITS: %d (桶数: %d)\n", RADIX_BITS, RADIX);
    printf("  COARSE_FACTOR: %d\n\n", COARSE_FACTOR);
    
    // 生成随机数据
    printf("生成随机测试数据...\n\n");
    unsigned int* h_original = new unsigned int[N];
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_original[i] = rand();
    }
    
    printf("=== 正确性验证 ===\n\n");
    
    // 1. 朴素三核基数排序
    testSort("1. 朴素三核基数排序 (书中图13.4)", gpuRadixSortNaive, h_original, N);
    
    // 2. 内存合并基数排序（练习1）
    testSort("2. 内存合并基数排序 (练习1: 共享内存优化)", gpuRadixSortCoalesced, h_original, N);
    
    // 3. 多位基数排序（练习2）
    testSort("3. 多位基数排序 (练习2: 4位/轮)", gpuRadixSortMultibitWrapper, h_original, N);
    
    // 4. 线程粗化基数排序（练习3）
    testSort("4. 线程粗化基数排序 (练习3: 每线程4元素)", gpuRadixSortCoarsenedWrapper, h_original, N);
    
    // 5. 并行归并排序（练习4）
    testSort("5. 并行归并排序 (练习4: 使用第12章归并)", gpuMergeSort, h_original, N);
    
    // 显示部分结果
    printf("=== 排序结果示例 ===\n");
    unsigned int* d_demo;
    unsigned int* h_demo = new unsigned int[N];
    cudaMalloc(&d_demo, N * sizeof(unsigned int));
    cudaMemcpy(d_demo, h_original, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    gpuRadixSortNaive(d_demo, N);
    cudaMemcpy(h_demo, d_demo, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("前10个元素: ");
    for (int i = 0; i < 10; i++) {
        printf("%u ", h_demo[i]);
    }
    printf("...\n");
    printf("后10个元素: ...");
    for (int i = N - 10; i < N; i++) {
        printf(" %u", h_demo[i]);
    }
    printf("\n\n");
    
    delete[] h_demo;
    cudaFree(d_demo);
    
    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• 基数排序：按位从低到高，利用前缀和确定输出位置\n");
    printf("• 内存合并：分开0/1的偏移计算，实现连续内存写入\n");
    printf("• 多位基数：每次处理多位（如4位），减少迭代轮数\n");
    printf("• 线程粗化：每线程处理多元素，提高寄存器利用率\n");
    printf("• 归并排序：Block内排序 + 跨Block归并，适合通用类型\n");
    printf("\n");
    
    // 清理
    delete[] h_original;
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
