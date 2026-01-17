/**
 * 第七章：卷积 - 3D卷积测试
 * 
 * 测试练习 8-10 的3D卷积实现
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-4f;

void initVolume(float* volume, int size) {
    for (int i = 0; i < size; ++i) {
        volume[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * 初始化3D高斯滤波器
 */
void init3DGaussianFilter(float* filter, int r) {
    int size = 2 * r + 1;
    float sigma = r / 2.0f;
    if (sigma == 0) sigma = 1.0f;
    float sum = 0.0f;
    
    for (int z = 0; z < size; ++z) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float dx = x - r;
                float dy = y - r;
                float dz = z - r;
                float value = expf(-(dx*dx + dy*dy + dz*dz) / (2 * sigma * sigma));
                filter[z * size * size + y * size + x] = value;
                sum += value;
            }
        }
    }
    
    // 归一化
    int totalSize = size * size * size;
    for (int i = 0; i < totalSize; ++i) {
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
    printf("  3D Convolution (练习 8-10)\n");
    printf("  参考: chapter-07 README.md\n");
    printf("================================================\n\n");
    
    // 测试参数 - 3D 使用较小的尺寸
    int width = 16;
    int height = 16;
    int depth = 16;
    int volumeSize = width * height * depth;
    int r = FILTER_RADIUS;
    int filterSize = 2 * r + 1;
    int filterElements = filterSize * filterSize * filterSize;
    
    printf("体积大小: %d × %d × %d\n", width, height, depth);
    printf("滤波器大小: %d × %d × %d (半径=%d)\n", filterSize, filterSize, filterSize, r);
    printf("IN_TILE_DIM: %d, OUT_TILE_DIM: %d\n\n", IN_TILE_DIM, OUT_TILE_DIM);
    
    // 分配内存
    float* N = new float[volumeSize];
    float* F = new float[filterElements];
    float* P_cpu = new float[volumeSize];
    float* P_basic = new float[volumeSize];
    float* P_const = new float[volumeSize];
    float* P_tiled = new float[volumeSize];
    
    // 初始化
    srand(42);
    initVolume(N, volumeSize);
    init3DGaussianFilter(F, r);
    
    printf("3D 高斯滤波器 (中心切片 z=%d):\n", r);
    print3DSlice(F, filterSize, filterSize, filterSize, r);
    printf("\n");
    
    // === CPU 参考实现 ===
    printf("=== CPU 参考实现 ===\n");
    conv3d_cpu(N, F, P_cpu, r, width, height, depth);
    printf("完成\n\n");
    
    // === 练习 8：基础 3D 卷积 ===
    printf("=== 练习 8：基础 3D 卷积 ===\n");
    conv3d_basic(N, F, P_basic, r, width, height, depth);
    if (verifyResults(P_cpu, P_basic, volumeSize)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // === 练习 9：常量内存 3D 卷积 ===
    printf("=== 练习 9：常量内存 3D 卷积 ===\n");
    conv3d_const_memory(N, F, P_const, r, width, height, depth);
    if (verifyResults(P_cpu, P_const, volumeSize)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // === 练习 10：Tiled 3D 卷积 ===
    printf("=== 练习 10：Tiled 3D 卷积 ===\n");
    conv3d_tiled(N, F, P_tiled, r, width, height, depth);
    if (verifyResults(P_cpu, P_tiled, volumeSize)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // 输出中心切片对比
    printf("================================================\n");
    printf("  输出对比 (中心切片 z=%d)\n", depth / 2);
    printf("================================================\n\n");
    
    printf("CPU 结果:\n");
    print3DSlice(P_cpu, width, height, depth, depth / 2);
    printf("\n");
    
    printf("【关键概念】\n");
    printf("------------------------------------------------\n");
    printf("• 3D 卷积：扩展2D卷积到三维空间\n");
    printf("• 滤波器大小：(2r+1)³ = %d 个元素\n", filterElements);
    printf("• 计算复杂度：每个输出需要 %d 次乘加\n", filterElements);
    printf("• Block 设计：使用较小的 block (8³=512 线程)\n");
    printf("• 共享内存：3D tile 需要更多内存\n");
    printf("\n");
    
    delete[] N;
    delete[] F;
    delete[] P_cpu;
    delete[] P_basic;
    delete[] P_const;
    delete[] P_tiled;
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
