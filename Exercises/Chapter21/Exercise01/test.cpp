#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "solution.h"
#include "../../../Common/timer.h"

// ============================================================================
// 辅助函数
// ============================================================================

// 生成随机 Bezier 曲线
void generateRandomCurves(BezierLine* lines, int num_lines, unsigned int seed) {
    srand(seed);
    
    for (int i = 0; i < num_lines; i++) {
        // 生成 3 个随机控制点
        for (int j = 0; j < 3; j++) {
            lines[i].CP[j * 2] = (float)rand() / RAND_MAX;       // x
            lines[i].CP[j * 2 + 1] = (float)rand() / RAND_MAX;   // y
        }
        lines[i].nVertices = 0;
    }
}

// 验证结果
bool verifyResults(BezierLine* result1, BezierLine* result2, int num_lines, float tolerance) {
    for (int i = 0; i < num_lines; i++) {
        if (result1[i].nVertices != result2[i].nVertices) {
            printf("曲线 %d: 顶点数不匹配 (%d vs %d)\n", 
                   i, result1[i].nVertices, result2[i].nVertices);
            return false;
        }
        
        for (int j = 0; j < result1[i].nVertices * 2; j++) {
            float diff = fabsf(result1[i].vertexPos[j] - result2[i].vertexPos[j]);
            if (diff > tolerance) {
                printf("曲线 %d, 位置 %d: 值不匹配 (%.6f vs %.6f, diff=%.6f)\n",
                       i, j, result1[i].vertexPos[j], result2[i].vertexPos[j], diff);
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  第二十一章：CUDA 动态并行性\n");
    printf("  Exercise 01: Bezier 曲线自适应细分\n");
    printf("================================================================\n\n");
    
    // 检查 CUDA 设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("错误：未找到 CUDA 设备\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (计算能力 %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("警告：动态并行需要计算能力 3.5+\n");
    }
    printf("\n");
    
    // 测试参数
    int num_lines = 1000;
    unsigned int seed = 42;
    
    printf("测试参数:\n");
    printf("  曲线数量: %d\n", num_lines);
    printf("  随机种子: %u\n\n", seed);
    
    // 分配内存
    BezierLine* lines_static = (BezierLine*)malloc(num_lines * sizeof(BezierLine));
    BezierLine* lines_dynamic = (BezierLine*)malloc(num_lines * sizeof(BezierLine));
    
    // 生成相同的随机曲线
    generateRandomCurves(lines_static, num_lines, seed);
    generateRandomCurves(lines_dynamic, num_lines, seed);
    
    // 测试静态版本
    printf("运行静态版本...\n");
    CudaTimer timer_static;
    timer_static.start();
    
    int ret = tessellate_bezier_static(lines_static, num_lines);
    if (ret != 0) {
        printf("静态版本失败，错误码: %d\n", ret);
        return 1;
    }
    
    timer_static.stop();
    float time_static = timer_static.elapsed_ms();
    printf("  静态版本耗时: %.3f ms\n", time_static);
    
    // 测试动态并行版本
    printf("运行动态并行版本...\n");
    CudaTimer timer_dynamic;
    timer_dynamic.start();
    
    ret = tessellate_bezier_dynamic(lines_dynamic, num_lines);
    if (ret != 0) {
        printf("动态并行版本失败，错误码: %d\n", ret);
        return 1;
    }
    
    timer_dynamic.stop();
    float time_dynamic = timer_dynamic.elapsed_ms();
    printf("  动态并行版本耗时: %.3f ms\n\n", time_dynamic);
    
    // 验证结果
    printf("验证结果...\n");
    bool match = verifyResults(lines_static, lines_dynamic, num_lines, 1e-5f);
    if (match) {
        printf("  ✓ 结果匹配！\n\n");
    } else {
        printf("  ✗ 结果不匹配！\n\n");
    }
    
    // 性能对比
    printf("================================================================\n");
    printf("性能对比:\n");
    printf("  静态版本:     %.3f ms\n", time_static);
    printf("  动态并行版本: %.3f ms\n", time_dynamic);
    printf("  加速比:       %.2fx\n", time_static / time_dynamic);
    printf("================================================================\n\n");
    
    // 打印一些样本结果
    printf("样本结果（前 5 条曲线）:\n");
    for (int i = 0; i < 5 && i < num_lines; i++) {
        printf("  曲线 %d: %d 顶点\n", i, lines_static[i].nVertices);
    }
    
    // 清理
    free(lines_static);
    free(lines_dynamic);
    
    return 0;
}
