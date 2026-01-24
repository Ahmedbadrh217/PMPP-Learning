#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#include "solution.h"
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 生成随机原子数据
 * @param atoms 输出数组 [numatoms * 4]，每个原子 (x, y, z, charge)
 * @param numatoms 原子数量
 * @param max_x, max_y, max_z 坐标范围
 */
void generate_atoms(float* atoms, int numatoms, float max_x, float max_y, float max_z) {
    srand(RANDOM_SEED);

    for (int i = 0; i < numatoms; i++) {
        atoms[i * 4 + 0] = (float)rand() / RAND_MAX * max_x;  // x
        atoms[i * 4 + 1] = (float)rand() / RAND_MAX * max_y;  // y
        atoms[i * 4 + 2] = (float)rand() / RAND_MAX * max_z;  // z
        atoms[i * 4 + 3] = ((float)rand() / RAND_MAX * 4.0f) - 2.0f;  // charge: -2 ~ 2
    }
}

/**
 * 比较两个能量网格是否接近
 * @return 1 表示匹配，0 表示不匹配
 */
int grids_allclose(const float* grid1, const float* grid2, dim3 grid_dimensions,
                   float rtol, float atol, int verbose) {
    int total_points = grid_dimensions.x * grid_dimensions.y * grid_dimensions.z;
    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int i = 0; i < total_points; i++) {
        float abs_diff = fabsf(grid1[i] - grid2[i]);
        float threshold = atol + rtol * fabsf(grid2[i]);

        if (abs_diff > threshold) {
            mismatches++;

            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_diff_idx = i;
            }

            if (verbose && mismatches <= 10) {
                int z = i / (grid_dimensions.x * grid_dimensions.y);
                int remainder = i % (grid_dimensions.x * grid_dimensions.y);
                int y = remainder / grid_dimensions.x;
                int x = remainder % grid_dimensions.x;
                printf("  不匹配位置 (%d, %d, %d): %.6f vs %.6f (差异: %.6f)\n",
                       x, y, z, grid1[i], grid2[i], abs_diff);
            }
        }
    }

    if (mismatches > 0) {
        printf("  发现 %d 个不匹配点 (共 %d 点, %.2f%%)\n",
               mismatches, total_points, (float)mismatches * 100.0f / total_points);
        return 0;
    }

    return 1;
}

// ============================================================================
// 正确性验证
// ============================================================================

void test_correctness() {
    printf("\n================================================================\n");
    printf("  正确性验证\n");
    printf("================================================================\n\n");

    // 小规模测试
    dim3 grid(64, 64, 32);
    float gridspacing = 0.5f;
    float z = 8.0f;
    int numatoms = 1000;

    printf("网格尺寸: %d x %d x %d\n", grid.x, grid.y, grid.z);
    printf("网格间距: %.2f Å\n", gridspacing);
    printf("原子数量: %d\n", numatoms);
    printf("测试 z 平面: %.1f\n\n", z);

    // 分配内存
    size_t grid_size = grid.x * grid.y * grid.z;
    float* atoms = (float*)malloc(numatoms * 4 * sizeof(float));
    float* energygrid_ref = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_test = (float*)calloc(grid_size, sizeof(float));

    if (!atoms || !energygrid_ref || !energygrid_test) {
        printf("内存分配失败！\n");
        return;
    }

    // 生成随机原子
    float max_x = grid.x * gridspacing;
    float max_y = grid.y * gridspacing;
    float max_z = grid.z * gridspacing;
    generate_atoms(atoms, numatoms, max_x, max_y, max_z);

    // CPU 参考结果
    printf("1. 计算 CPU 参考结果...\n");
    cenergySequential(energygrid_ref, grid, gridspacing, z, atoms, numatoms);

    // 测试各 GPU 实现
    const char* impl_names[] = {
        "GPU Scatter (Fig. 18.5)",
        "GPU Gather (Fig. 18.6)",
        "GPU Thread Coarsening (Fig. 18.8)",
        "GPU Memory Coalescing (Fig. 18.10)"
    };

    void (*impl_funcs[])(float*, dim3, float, float, const float*, int) = {
        cenergyParallelScatter,
        cenergyParallelGather,
        cenergyParallelCoarsen,
        cenergyParallelCoalescing
    };

    int num_impls = sizeof(impl_funcs) / sizeof(impl_funcs[0]);
    int all_passed = 1;

    for (int i = 0; i < num_impls; i++) {
        printf("\n%d. 测试 %s...\n", i + 2, impl_names[i]);
        memset(energygrid_test, 0, grid_size * sizeof(float));

        impl_funcs[i](energygrid_test, grid, gridspacing, z, atoms, numatoms);

        if (grids_allclose(energygrid_test, energygrid_ref, grid, 1e-2f, 1e-3f, 1)) {
            printf("   ✅ 正确！\n");
        } else {
            printf("   ❌ 失败！\n");
            all_passed = 0;
        }
    }

    if (all_passed) {
        printf("\n所有实现通过正确性验证！✅\n");
    }

    free(atoms);
    free(energygrid_ref);
    free(energygrid_test);
}

// ============================================================================
// 性能基准测试
// ============================================================================

void test_performance() {
    printf("\n================================================================\n");
    printf("  性能基准测试\n");
    printf("================================================================\n\n");

    // 较大规模测试
    dim3 grid(128, 128, 64);
    float gridspacing = 0.5f;
    float z = 16.0f;
    int numatoms = 5000;

    printf("网格尺寸: %d x %d x %d\n", grid.x, grid.y, grid.z);
    printf("网格间距: %.2f Å\n", gridspacing);
    printf("原子数量: %d\n", numatoms);
    printf("测试 z 平面: %.1f\n", z);

    int warmup_runs = 3;
    int timing_runs = 10;
    printf("预热次数: %d, 计时次数: %d\n\n", warmup_runs, timing_runs);

    // 分配内存
    size_t grid_size = grid.x * grid.y * grid.z;
    float* atoms = (float*)malloc(numatoms * 4 * sizeof(float));
    float* energygrid = (float*)calloc(grid_size, sizeof(float));

    if (!atoms || !energygrid) {
        printf("内存分配失败！\n");
        return;
    }

    // 生成随机原子
    float max_x = grid.x * gridspacing;
    float max_y = grid.y * gridspacing;
    float max_z = grid.z * gridspacing;
    generate_atoms(atoms, numatoms, max_x, max_y, max_z);

    // CPU Sequential 基准
    printf("1. CPU Sequential...\n");
    Timer cpu_timer;
    for (int i = 0; i < warmup_runs; i++) {
        cenergySequential(energygrid, grid, gridspacing, z, atoms, numatoms);
    }
    cpu_timer.start();
    for (int i = 0; i < timing_runs; i++) {
        cenergySequential(energygrid, grid, gridspacing, z, atoms, numatoms);
    }
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed_ms() / timing_runs;
    printf("   时间: %.3f ms\n", cpu_time);

    // GPU 实现基准
    const char* impl_names[] = {
        "GPU Scatter (Fig. 18.5)",
        "GPU Gather (Fig. 18.6)",
        "GPU Coarsen (Fig. 18.8)",
        "GPU Coalescing (Fig. 18.10)"
    };

    void (*impl_funcs[])(float*, dim3, float, float, const float*, int) = {
        cenergyParallelScatter,
        cenergyParallelGather,
        cenergyParallelCoarsen,
        cenergyParallelCoalescing
    };

    int num_impls = sizeof(impl_funcs) / sizeof(impl_funcs[0]);
    float gpu_times[4];

    for (int i = 0; i < num_impls; i++) {
        printf("\n%d. %s...\n", i + 2, impl_names[i]);

        // 预热
        for (int w = 0; w < warmup_runs; w++) {
            impl_funcs[i](energygrid, grid, gridspacing, z, atoms, numatoms);
        }

        // 计时
        CudaTimer cuda_timer;
        cuda_timer.start();
        for (int t = 0; t < timing_runs; t++) {
            impl_funcs[i](energygrid, grid, gridspacing, z, atoms, numatoms);
        }
        cuda_timer.stop();

        gpu_times[i] = cuda_timer.elapsed_ms() / timing_runs;
        float speedup = cpu_time / gpu_times[i];
        printf("   时间: %.3f ms (加速比: %.2fx)\n", gpu_times[i], speedup);
    }

    // 性能总结
    printf("\n================================================================\n");
    printf("  性能总结\n");
    printf("================================================================\n\n");

    printf("| 实现 | 时间 (ms) | 相对 CPU 加速 |\n");
    printf("|------|-----------|---------------|\n");
    printf("| CPU Sequential | %.3f | 1.00x |\n", cpu_time);
    for (int i = 0; i < num_impls; i++) {
        printf("| %s | %.3f | %.2fx |\n",
               impl_names[i], gpu_times[i], cpu_time / gpu_times[i]);
    }

    free(atoms);
    free(energygrid);
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  第十八章：静电势能图\n");
    printf("  Electrostatic Potential Map - Multiple Implementations\n");
    printf("================================================================\n");

    // 打印设备信息
    printDeviceInfo();

    // 正确性验证
    test_correctness();

    // 性能基准测试
    test_performance();

    printf("\n测试完成！\n");
    return 0;
}
