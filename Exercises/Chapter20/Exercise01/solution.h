#ifndef CHAPTER_20_SOLUTION_H
#define CHAPTER_20_SOLUTION_H

#include <cuda_runtime.h>
#include <mpi.h>

// ============================================================================
// 常量定义
// ============================================================================

#define DATA_COLLECT 100    // MPI 消息标签：数据收集
#define HALO_SIZE 4         // Halo 大小（模板半径）
#define BLOCK_DIM_X 8       // 线程块 X 维度
#define BLOCK_DIM_Y 8       // 线程块 Y 维度
#define BLOCK_DIM_Z 8       // 线程块 Z 维度

// ============================================================================
// 辅助函数声明
// ============================================================================

/**
 * 初始化随机数据
 */
void random_data(float* data, int dimx, int dimy, int dimz, float min_val, float max_val);

/**
 * 存储输出数据（调试用）
 */
void store_output(float* output, int dimx, int dimy, int dimz);

/**
 * 上传模板系数到常量内存
 */
void upload_coefficients(float* host_coeff, int num_coeff);

// ============================================================================
// 核心函数声明
// ============================================================================

/**
 * 调用模板计算 CUDA 核函数
 * @param d_output 输出数据（设备内存）
 * @param d_input  输入数据（设备内存）
 * @param dimx     X 维度
 * @param dimy     Y 维度
 * @param dimz     Z 维度
 * @param stream   CUDA 流
 */
void call_stencil_kernel(float* d_output, float* d_input,
                         int dimx, int dimy, int dimz, cudaStream_t stream);

/**
 * 数据服务器进程
 * - 分发输入数据给计算节点
 * - 收集计算结果
 * @param dimx   X 维度
 * @param dimy   Y 维度
 * @param dimz   Z 维度
 * @param nreps  迭代次数
 */
void data_server(int dimx, int dimy, int dimz, int nreps);

/**
 * 计算节点进程
 * - 接收数据
 * - 执行模板计算
 * - Halo 交换
 * - 发送结果
 * @param dimx   X 维度
 * @param dimy   Y 维度
 * @param dimz   本节点 Z 维度
 * @param nreps  迭代次数
 */
void compute_node_stencil(int dimx, int dimy, int dimz, int nreps);

#endif // CHAPTER_20_SOLUTION_H
