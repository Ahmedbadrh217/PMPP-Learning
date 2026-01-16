/**
 * 第五章：内存架构和数据局部性 - 动态 Tile 大小矩阵乘法
 * 
 * 参考：chapter-05/code/matrix_mul_with_optimal_dynamic_tile_size.cu
 * 
 * 本程序演示如何根据硬件规格动态计算最优 Tile 大小，
 * 而不是使用硬编码的固定值。
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../../Common/timer.h"
#include "solution.h"

const float TOLERANCE = 1e-3f;

void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

double benchmark(void (*func)(float*, const float*, const float*, int, int, int),
                 float* P, const float* M, const float* N, 
                 int m, int n, int o, int iterations) {
    // 预热
    func(P, M, N, m, n, o);
    
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func(P, M, N, m, n, o);
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}

int main() {
    printf("\n");
    printf("================================================\n");
    printf("  第五章：内存架构和数据局部性\n");
    printf("  Dynamic Tile Size Matrix Multiplication\n");
    printf("  参考: matrix_mul_with_optimal_dynamic_tile_size.cu\n");
    printf("================================================\n\n");

    // 使用非方阵测试动态 Tile 的适应性
    const int M_ROWS = 1024;
    const int M_COLS = 1536;
    const int N_COLS = 2048;
    const int ITERATIONS = 10;

    printf("矩阵大小: %d × %d × %d (非方阵)\n", M_ROWS, M_COLS, N_COLS);
    printf("测试迭代次数: %d\n\n", ITERATIONS);

    // 分配内存
    float* h_M = new float[M_ROWS * M_COLS];
    float* h_N = new float[M_COLS * N_COLS];
    float* h_P_naive = new float[M_ROWS * N_COLS];
    float* h_P_dynamic = new float[M_ROWS * N_COLS];

    // 初始化
    srand(42);
    initMatrix(h_M, M_ROWS, M_COLS);
    initMatrix(h_N, M_COLS, N_COLS);

    printf("=== 正确性验证 ===\n");
    
    matrixMul(h_P_naive, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    matrixMulTilingDynamic(h_P_dynamic, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    
    if (verifyResults(h_P_naive, h_P_dynamic, M_ROWS * N_COLS)) {
        printf("✅ 结果一致！\n\n");
    } else {
        printf("❌ 结果不一致！\n\n");
    }

    printf("=== 性能测试 ===\n");
    
    double naiveTime = benchmark(matrixMul, h_P_naive, h_M, h_N, 
                                  M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("朴素矩阵乘法:        %.3f ms\n", naiveTime);
    
    double dynamicTime = benchmark(matrixMulTilingDynamic, h_P_dynamic, h_M, h_N, 
                                    M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("动态 Tile 矩阵乘法:  %.3f ms (%.2fx)\n", dynamicTime, naiveTime / dynamicTime);

    printf("\n");
    printf("【关键概念】\n");
    printf("------------------------------------------------\n");
    printf("• 动态 Tile 大小根据硬件规格计算\n");
    printf("• 使用 extern __shared__ 动态分配共享内存\n");
    printf("• kernel 启动时指定共享内存大小\n");
    printf("• 适应不同 GPU 和矩阵大小\n");
    printf("\n");

    delete[] h_M;
    delete[] h_N;
    delete[] h_P_naive;
    delete[] h_P_dynamic;

    printf("✅ 测试完成！\n\n");
    return 0;
}
