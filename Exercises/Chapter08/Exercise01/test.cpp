/**
 * 第八章：模板 - 3D模板测试
 * 
 * 参考：chapter-08/code/benchmark.cu
 * 
 * 测试所有5种实现的正确性
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

bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    int mismatchCount = 0;
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            if (mismatchCount < 5) {
                printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n", 
                       i, A[i], B[i], fabs(A[i] - B[i]));
            }
            mismatchCount++;
        }
    }
    if (mismatchCount > 0) {
        printf("Total mismatches: %d\n", mismatchCount);
    }
    return mismatchCount == 0;
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第八章：模板\n");
    printf("  3D Stencil Operations - 5 Implementations\n");
    printf("  参考: chapter-08/code/stencil.cu\n");
    printf("================================================================\n\n");
    
    // 测试参数
    unsigned int N = 64;  // 网格大小
    size_t volumeSize = N * N * N;
    
    // 系数
    int c0 = 0, c1 = 1, c2 = 1, c3 = 1, c4 = 1, c5 = 1, c6 = 1;
    
    printf("配置:\n");
    printf("  网格大小: %d × %d × %d\n", N, N, N);
    printf("  总元素数: %zu\n", volumeSize);
    printf("  内存大小: %.2f MB\n", volumeSize * sizeof(float) / (1024.0f * 1024.0f));
    printf("  OUT_TILE_DIM_SMALL: %d, IN_TILE_DIM_SMALL: %d\n", OUT_TILE_DIM_SMALL, IN_TILE_DIM_SMALL);
    printf("  OUT_TILE_DIM_BIG: %d, IN_TILE_DIM_BIG: %d\n\n", OUT_TILE_DIM_BIG, IN_TILE_DIM_BIG);
    
    // 分配内存
    float* in = new float[volumeSize];
    float* out_seq = new float[volumeSize];
    float* out_basic = new float[volumeSize];
    float* out_shared = new float[volumeSize];
    float* out_coarse = new float[volumeSize];
    float* out_register = new float[volumeSize];
    
    // 初始化
    srand(42);
    initVolume(in, volumeSize);
    memset(out_seq, 0, volumeSize * sizeof(float));
    memset(out_basic, 0, volumeSize * sizeof(float));
    memset(out_shared, 0, volumeSize * sizeof(float));
    memset(out_coarse, 0, volumeSize * sizeof(float));
    memset(out_register, 0, volumeSize * sizeof(float));
    
    printf("=== 正确性验证 ===\n\n");
    
    // 1. CPU 顺序实现（参考）
    printf("1. CPU 顺序实现...\n");
    stencil_3d_sequential(in, out_seq, N, c0, c1, c2, c3, c4, c5, c6);
    printf("   完成\n\n");
    
    // 2. 基础并行
    printf("2. 基础并行 (图8.6)...\n");
    stencil_3d_parallel_basic(in, out_basic, N, c0, c1, c2, c3, c4, c5, c6);
    if (verifyResults(out_seq, out_basic, volumeSize)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }
    
    // 3. 共享内存
    printf("3. 共享内存 (图8.8)...\n");
    stencil_3d_parallel_shared_memory(in, out_shared, N, c0, c1, c2, c3, c4, c5, c6);
    if (verifyResults(out_seq, out_shared, volumeSize)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }
    
    // 4. 线程粗化
    printf("4. 线程粗化 (图8.10)...\n");
    stencil_3d_parallel_thread_coarsening(in, out_coarse, N, c0, c1, c2, c3, c4, c5, c6);
    if (verifyResults(out_seq, out_coarse, volumeSize)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }
    
    // 5. 寄存器优化
    printf("5. 寄存器优化 (图8.12)...\n");
    stencil_3d_parallel_register_tiling(in, out_register, N, c0, c1, c2, c3, c4, c5, c6);
    if (verifyResults(out_seq, out_register, volumeSize)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }
    
    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• 七点模板：中心 + 上下前后左右\n");
    printf("• 共享内存 Tiling：3D tile 减少全局内存访问\n");
    printf("• 线程粗化：Z方向遍历，减少边界线程浪费\n");
    printf("• 寄存器优化：Z方向存寄存器，XY平面存共享内存\n");
    printf("• 滑动窗口：prev/curr/next 三层循环复用\n");
    printf("\n");
    
    // 释放内存
    delete[] in;
    delete[] out_seq;
    delete[] out_basic;
    delete[] out_shared;
    delete[] out_coarse;
    delete[] out_register;
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
