#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第八章：模板 - 3D七点模板实现
 * 
 * 包含5种实现：
 * 1. 顺序实现（CPU参考）
 * 2. 基础并行（图8.6）
 * 3. 共享内存 Tiling（图8.8）
 * 4. 线程粗化（图8.10）
 * 5. 寄存器优化（图8.12）
 */

#include <cuda_runtime.h>

// Block 尺寸定义
// 部分 kernel 需要 3D 共享内存（立方），部分只需 2D（平方）
#define OUT_TILE_DIM_SMALL 8
#define IN_TILE_DIM_SMALL (OUT_TILE_DIM_SMALL + 2)  // 10
#define OUT_TILE_DIM_BIG 30
#define IN_TILE_DIM_BIG (OUT_TILE_DIM_BIG + 2)      // 32

// 全局系数
extern int c0, c1, c2, c3, c4, c5, c6;

// 辅助函数
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * CPU 顺序实现（参考）
 */
void stencil_3d_sequential(float* in, float* out, unsigned int N,
                           int c0, int c1, int c2, int c3, int c4, int c5, int c6);

/**
 * 基础并行实现（图8.6）
 * 每个线程计算一个输出点，无共享内存优化
 */
void stencil_3d_parallel_basic(float* in, float* out, unsigned int N,
                               int c0, int c1, int c2, int c3, int c4, int c5, int c6);

/**
 * 共享内存实现（图8.8）
 * 使用 3D 共享内存 Tile
 */
void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N,
                                       int c0, int c1, int c2, int c3, int c4, int c5, int c6);

/**
 * 线程粗化实现（图8.10）
 * Z方向粗化，使用三层共享内存滑动窗口
 */
void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6);

/**
 * 寄存器优化实现（图8.12）
 * Z方向数据存寄存器，XY平面存共享内存
 */
void stencil_3d_parallel_register_tiling(float* in, float* out, unsigned int N,
                                         int c0, int c1, int c2, int c3, int c4, int c5, int c6);

/**
 * 打印3D数组切片（调试用）
 */
void print3DSlice(float* arr, int N, int z);

#endif // SOLUTION_H
