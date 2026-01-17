/**
 * 第七章：卷积 - Tiled 2D卷积测试
 * 
 * 测试 Tiled 卷积和 L2 缓存版本的正确性和性能
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-4f;

void initMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void initGaussianFilter(float* filter, int r) {
    int size = 2 * r + 1;
    float sigma = r / 2.0f;
    if (sigma == 0) sigma = 1.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - r;
            float y = j - r;
            float value = expf(-(x*x + y*y) / (2 * sigma * sigma));
            filter[i * size + j] = value;
            sum += value;
        }
    }
    
    for (int i = 0; i < size * size; ++i) {
        filter[i] /= sum;
    }
}

bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n", 
                   i, A[i], B[i], fabs(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

int main() {
    printf("\n");
    printf("================================================\n");
    printf("  第七章：卷积\n");
    printf("  Tiled 2D Convolution\n");
    printf("  参考: chapter-07/code (Fig 7.12, 7.15)\n");
    printf("================================================\n\n");
    
    // 测试参数
    int height = 128;
    int width = 128;
    int r = FILTER_RADIUS;
    int filterSize = 2 * r + 1;
    
    printf("矩阵大小: %d × %d\n", height, width);
    printf("滤波器大小: %d × %d (半径=%d)\n", filterSize, filterSize, r);
    printf("IN_TILE_SIZE: %d, OUT_TILE_SIZE: %d\n\n", IN_TILE_SIZE, OUT_TILE_SIZE);
    
    // 分配内存
    float* N = new float[height * width];
    float* F = new float[filterSize * filterSize];
    float* P_cpu = new float[height * width];
    float* P_tiled = new float[height * width];
    float* P_cached = new float[height * width];
    
    // 初始化
    srand(42);
    initMatrix(N, height * width);
    initGaussianFilter(F, r);
    
    printf("高斯滤波器:\n");
    printMatrix(F, filterSize, filterSize);
    printf("\n");
    
    // === CPU 参考实现 ===
    printf("=== CPU 参考实现 ===\n");
    conv2d_cpu(N, F, P_cpu, r, height, width);
    printf("完成\n\n");
    
    // === Tiled GPU 实现 ===
    printf("=== Tiled GPU 实现 (图7.12) ===\n");
    conv2d_tiled(N, F, P_tiled, r, height, width);
    if (verifyResults(P_cpu, P_tiled, height * width)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // === Tiled + L2 缓存 GPU 实现 ===
    printf("=== Tiled + L2缓存 GPU 实现 (图7.15) ===\n");
    conv2d_tiled_cached(N, F, P_cached, r, height, width);
    if (verifyResults(P_cpu, P_cached, height * width)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    printf("【关键概念】\n");
    printf("------------------------------------------------\n");
    printf("• Tiled 卷积：共享内存存储整个输入 tile（含 halo）\n");
    printf("• 输入 tile 大小: %d × %d\n", IN_TILE_SIZE, IN_TILE_SIZE);
    printf("• 输出 tile 大小: %d × %d\n", OUT_TILE_SIZE, OUT_TILE_SIZE);
    printf("• L2 缓存版本：边界元素依赖 L2 缓存命中\n");
    printf("  - 共享内存使用减少\n");
    printf("  - 适合小滤波器场景\n");
    printf("\n");
    
    delete[] N;
    delete[] F;
    delete[] P_cpu;
    delete[] P_tiled;
    delete[] P_cached;
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
