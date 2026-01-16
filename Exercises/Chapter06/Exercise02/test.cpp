/**
 * 第六章：性能方面的考虑 - Thread Coarsening 矩阵乘法性能测试
 * 
 * 本程序对比三种矩阵乘法实现的性能：
 * 1. 朴素版本：直接全局内存访问
 * 2. Tiled 版本：使用共享内存
 * 3. Tiled + Thread Coarsening：每个线程计算多个输出元素
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../../Common/timer.h"
#include "solution.h"

// 结果验证容差
const float TOLERANCE = 1e-3f;

/**
 * 初始化矩阵（随机值）
 */
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * 验证两个矩阵是否近似相等
 */
bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

/**
 * 基准测试函数
 */
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
    printf("========================================\n");
    printf("  第六章：性能方面的考虑\n");
    printf("  Thread Coarsening Matrix Multiplication\n");
    printf("========================================\n\n");

    // 矩阵大小
    const int M_ROWS = 1024;
    const int M_COLS = 1024;
    const int N_COLS = 1024;
    const int ITERATIONS = 10;

    printf("矩阵大小: %d × %d × %d\n", M_ROWS, M_COLS, N_COLS);
    printf("测试迭代次数: %d\n", ITERATIONS);
    printf("Thread Coarsening Factor: 4\n\n");

    // 分配内存
    float* h_M = new float[M_ROWS * M_COLS];
    float* h_N = new float[M_COLS * N_COLS];
    float* h_P_naive = new float[M_ROWS * N_COLS];
    float* h_P_tiled = new float[M_ROWS * N_COLS];
    float* h_P_coarsened = new float[M_ROWS * N_COLS];

    // 初始化矩阵
    srand(42);
    initMatrix(h_M, M_ROWS, M_COLS);
    initMatrix(h_N, M_COLS, N_COLS);

    printf("=== 正确性验证 ===\n");
    
    // 运行三种版本
    matrixMul(h_P_naive, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    matrixMulTiled(h_P_tiled, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    matrixMulTiledCoarsened(h_P_coarsened, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    
    // 验证结果
    bool tiled_ok = verifyResults(h_P_naive, h_P_tiled, M_ROWS * N_COLS);
    bool coarsened_ok = verifyResults(h_P_naive, h_P_coarsened, M_ROWS * N_COLS);
    
    if (tiled_ok && coarsened_ok) {
        printf("✅ 所有方法结果一致！\n\n");
    } else {
        if (!tiled_ok) printf("❌ Tiled 版本结果不一致！\n");
        if (!coarsened_ok) printf("❌ Coarsened 版本结果不一致！\n");
        printf("\n");
    }

    printf("=== 性能测试 ===\n");
    
    // 朴素版本性能
    double naiveTime = benchmark(matrixMul, h_P_naive, h_M, h_N, 
                                  M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("朴素矩阵乘法:          %.3f ms\n", naiveTime);
    
    // Tiled 版本性能
    double tiledTime = benchmark(matrixMulTiled, h_P_tiled, h_M, h_N, 
                                  M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("Tiled 矩阵乘法:        %.3f ms (%.2fx vs naive)\n", 
           tiledTime, naiveTime / tiledTime);
    
    // Thread Coarsening 版本性能
    double coarsenedTime = benchmark(matrixMulTiledCoarsened, h_P_coarsened, h_M, h_N, 
                                      M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("Tiled + Coarsening:    %.3f ms (%.2fx vs naive, %.2fx vs tiled)\n", 
           coarsenedTime, naiveTime / coarsenedTime, tiledTime / coarsenedTime);

    // 计算 GFLOPS
    double gflops = 2.0 * M_ROWS * M_COLS * N_COLS / 1e9;
    printf("\n计算量: %.2f GFLOP\n", gflops);
    printf("朴素版本吞吐量:        %.2f GFLOPS\n", gflops / (naiveTime / 1000.0));
    printf("Tiled 版本吞吐量:      %.2f GFLOPS\n", gflops / (tiledTime / 1000.0));
    printf("Coarsening 版本吞吐量: %.2f GFLOPS\n", gflops / (coarsenedTime / 1000.0));

    printf("\n");
    printf("【关键概念】\n");
    printf("----------------------------------------\n");
    printf("• Thread Coarsening: 每个线程计算多个输出元素\n");
    printf("• 优势: M 矩阵 Tile 被复用多次，减少加载\n");
    printf("• 算术强度:\n");
    printf("  - 朴素版本:     0.25 OP/B\n");
    printf("  - Tiled (32×32): 8 OP/B\n");
    printf("  - Coarsening×4: 12.8 OP/B\n");
    printf("\n");

    // 清理
    delete[] h_M;
    delete[] h_N;
    delete[] h_P_naive;
    delete[] h_P_tiled;
    delete[] h_P_coarsened;

    printf("✅ 测试完成！\n\n");
    return 0;
}
