#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "solution.h"
#include "../../../Common/timer.h"

// ============================================================================
// 辅助函数
// ============================================================================

// 生成随机点
void generateRandomPoints(float* x, float* y, int num_points, unsigned int seed) {
    srand(seed);
    
    for (int i = 0; i < num_points; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = (float)rand() / RAND_MAX;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  第二十一章：CUDA 动态并行性\n");
    printf("  Exercise 02: 四叉树递归构建\n");
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
    int num_points = 1000;
    int max_depth = 5;
    int min_points = 4;
    unsigned int seed = 42;
    
    printf("测试参数:\n");
    printf("  点数量: %d\n", num_points);
    printf("  最大深度: %d\n", max_depth);
    printf("  节点最小点数: %d\n", min_points);
    printf("  随机种子: %u\n\n", seed);
    
    // 分配内存
    float* h_x = (float*)malloc(num_points * sizeof(float));
    float* h_y = (float*)malloc(num_points * sizeof(float));
    
    // 生成随机点
    generateRandomPoints(h_x, h_y, num_points, seed);
    
    // 边界框
    float bounds[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    
    // 结果
    float* result_x = NULL;
    float* result_y = NULL;
    int num_result_points = 0;
    
    // 构建四叉树
    printf("构建四叉树...\n");
    CudaTimer timer;
    timer.start();
    
    int ret = build_quadtree(h_x, h_y, num_points, max_depth, min_points,
                              &result_x, &result_y, bounds, &num_result_points);
    
    timer.stop();
    
    if (ret != 0) {
        printf("四叉树构建失败，错误码: %d\n", ret);
        return 1;
    }
    
    printf("  ✓ 四叉树构建完成\n");
    printf("  耗时: %.3f ms\n\n", timer.elapsed_ms());
    
    // 统计
    printf("================================================================\n");
    printf("结果统计:\n");
    printf("  输入点数: %d\n", num_points);
    printf("  输出点数: %d\n", num_result_points);
    printf("  最大深度: %d\n", max_depth);
    
    // 计算理论最大子 kernel 启动数
    int max_launches = 0;
    for (int d = 0; d < max_depth; d++) {
        max_launches += (int)pow(4, d);
    }
    printf("  理论最大子 kernel 数: %d\n", max_launches);
    printf("================================================================\n\n");
    
    // 打印一些样本点
    printf("样本点（前 5 个）:\n");
    for (int i = 0; i < 5 && i < num_points; i++) {
        printf("  点 %d: (%.4f, %.4f) -> (%.4f, %.4f)\n", 
               i, h_x[i], h_y[i], result_x[i], result_y[i]);
    }
    
    // 清理
    free(h_x);
    free(h_y);
    free(result_x);
    free(result_y);
    
    return 0;
}
