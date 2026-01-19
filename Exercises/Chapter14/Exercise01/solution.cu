/**
 * 第十四章：稀疏矩阵计算 - CUDA 实现
 * 
 * 参考：chapter-14/code/
 * 
 * 本实现包含：
 * 1. COO SpMV - 使用原子操作
 * 2. CSR SpMV - 每行一个线程
 * 3. ELL SpMV - 列主序，合并访问
 * 4. JDS SpMV - 练习5
 * 5. ELL-COO Hybrid SpMV - 练习4
 * 6. COO to CSR 转换 - 练习3
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ====================== 1. COO SpMV ======================

/**
 * COO SpMV Kernel
 * 每个线程处理一个非零元素，使用原子操作累加到对应行
 */
__global__ void spmv_coo_kernel(int nnz, const int* rowIdx, const int* colIdx, 
                                 const float* values, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float val = values[i];
        atomicAdd(&y[row], val * x[col]);
    }
}

/**
 * COO 格式 SpMV 主机接口
 */
void spmv_coo(const COOMatrix& A, const float* d_x, float* d_y) {
    // 清零输出向量
    CHECK_CUDA(cudaMemset(d_y, 0, A.numRows * sizeof(float)));
    
    int numBlocks = (A.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_coo_kernel<<<numBlocks, BLOCK_SIZE>>>(A.nnz, A.rowIdx, A.colIdx, A.values, d_x, d_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ====================== 2. CSR SpMV ======================

/**
 * CSR SpMV Kernel（每行一个线程）
 * 书中图14.6
 */
__global__ void spmv_csr_kernel(int numRows, const int* rowPtrs, const int* colIdx,
                                 const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        float sum = 0.0f;
        int rowStart = rowPtrs[row];
        int rowEnd = rowPtrs[row + 1];
        
        for (int j = rowStart; j < rowEnd; j++) {
            int col = colIdx[j];
            float val = values[j];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

/**
 * CSR 格式 SpMV 主机接口
 */
void spmv_csr(const CSRMatrix& A, const float* d_x, float* d_y) {
    int numBlocks = (A.numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_csr_kernel<<<numBlocks, BLOCK_SIZE>>>(A.numRows, A.rowPtrs, A.colIdx, A.values, d_x, d_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ====================== 3. ELL SpMV ======================

/**
 * ELL SpMV Kernel
 * 书中图14.10
 * 列主序存储：idx[t][row] = values[t * numRows + row]
 */
__global__ void spmv_ell_kernel(int numRows, int maxNnzPerRow, const int* colIdx,
                                 const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < maxNnzPerRow; t++) {
            int idx = t * numRows + row;  // 列主序索引
            int col = colIdx[idx];
            if (col >= 0) {  // 有效元素（-1 为填充）
                float val = values[idx];
                sum += val * x[col];
            }
        }
        y[row] = sum;
    }
}

/**
 * ELL 格式 SpMV 主机接口
 */
void spmv_ell(const ELLMatrix& A, const float* d_x, float* d_y) {
    int numBlocks = (A.numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_ell_kernel<<<numBlocks, BLOCK_SIZE>>>(A.numRows, A.maxNnzPerRow, A.colIdx, A.values, d_x, d_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ====================== 4. JDS SpMV（练习5） ======================

/**
 * JDS SpMV Kernel
 * 按行长度降序排列，减少填充浪费
 */
__global__ void spmv_jds_kernel(int numRows, int numTiles, const int* colIdx,
                                 const float* values, const int* rowPerm, 
                                 const int* iterPtr, const float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    
    float sum = 0.0f;
    
    // 遍历每个"对角线"（迭代）
    for (int t = 0; t < numTiles; t++) {
        int idx = iterPtr[t] + tid;
        // 检查是否在当前对角线的有效范围内
        if (idx < iterPtr[t + 1]) {
            int col = colIdx[idx];
            float val = values[idx];
            sum += val * x[col];
        }
    }
    
    // 按原始行顺序写回结果
    y[rowPerm[tid]] = sum;
}

/**
 * JDS 格式 SpMV 主机接口
 */
void spmv_jds(const JDSMatrix& A, const float* d_x, float* d_y) {
    int numBlocks = (A.numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_jds_kernel<<<numBlocks, BLOCK_SIZE>>>(A.numRows, A.numTiles, A.colIdx, 
                                                A.values, A.rowPerm, A.iterPtr, d_x, d_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ====================== 5. ELL-COO Hybrid SpMV（练习4） ======================

/**
 * Hybrid SpMV
 * ELL 部分在 GPU 上执行，COO 溢出部分在 CPU 上执行
 */
void spmv_hybrid(const ELLMatrix& ellPart, const COOMatrix& cooPart, 
                 const float* d_x, float* d_y) {
    // 1. ELL 部分：GPU 执行
    CHECK_CUDA(cudaMemset(d_y, 0, ellPart.numRows * sizeof(float)));
    
    int numBlocks = (ellPart.numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_ell_kernel<<<numBlocks, BLOCK_SIZE>>>(ellPart.numRows, ellPart.maxNnzPerRow,
                                                ellPart.colIdx, ellPart.values, d_x, d_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 2. COO 部分：拷贝回 CPU 执行
    if (cooPart.nnz > 0) {
        // 拷贝部分结果和向量到主机
        float* h_y = new float[ellPart.numRows];
        float* h_x = new float[ellPart.numCols];
        CHECK_CUDA(cudaMemcpy(h_y, d_y, ellPart.numRows * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_x, d_x, ellPart.numCols * sizeof(float), cudaMemcpyDeviceToHost));
        
        // CPU 上处理 COO 溢出部分
        for (int i = 0; i < cooPart.nnz; i++) {
            int row = cooPart.rowIdx[i];
            int col = cooPart.colIdx[i];
            float val = cooPart.values[i];
            h_y[row] += val * h_x[col];
        }
        
        // 拷贝回 GPU
        CHECK_CUDA(cudaMemcpy(d_y, h_y, ellPart.numRows * sizeof(float), cudaMemcpyHostToDevice));
        
        delete[] h_y;
        delete[] h_x;
    }
}

// ====================== 6. COO to CSR 转换（练习3） ======================

/**
 * Kernel: 计算直方图（每行的非零元素数）
 */
__global__ void computeHistogramKernel(int nnz, const int* rowIdx, int* rowPtrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&rowPtrs[rowIdx[i] + 1], 1);
    }
}

/**
 * Kernel: 前缀和（简化版，单 Block）
 */
__global__ void exclusiveScanKernel(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    if (tid < n) {
        temp[tid] = data[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Kogge-Stone 前缀和
    for (int offset = 1; offset < n; offset *= 2) {
        int val = 0;
        if (tid >= offset && tid < n) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid < n) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    if (tid < n) {
        data[tid] = temp[tid];
    }
}

/**
 * COO 到 CSR 格式转换
 * 使用并行直方图和前缀和
 */
void coo_to_csr(const COOMatrix& coo, CSRMatrix& csr) {
    int nnz = coo.nnz;
    int numRows = coo.numRows;
    
    // 分配 CSR 结构
    csr.numRows = numRows;
    csr.numCols = coo.numCols;
    csr.nnz = nnz;
    
    CHECK_CUDA(cudaMalloc(&csr.rowPtrs, (numRows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr.colIdx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr.values, nnz * sizeof(float)));
    
    // 初始化 rowPtrs 为 0
    CHECK_CUDA(cudaMemset(csr.rowPtrs, 0, (numRows + 1) * sizeof(int)));
    
    // 步骤1：计算直方图
    int numBlocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeHistogramKernel<<<numBlocks, BLOCK_SIZE>>>(nnz, coo.rowIdx, csr.rowPtrs);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 步骤2：前缀和
    // 注意：这里使用简化版单 Block 前缀和，适用于小规模矩阵
    if (numRows + 1 <= 1024) {
        exclusiveScanKernel<<<1, numRows + 1, (numRows + 1) * sizeof(int)>>>(csr.rowPtrs, numRows + 1);
    } else {
        // 对于大规模矩阵，在 CPU 上执行前缀和
        int* h_rowPtrs = new int[numRows + 1];
        CHECK_CUDA(cudaMemcpy(h_rowPtrs, csr.rowPtrs, (numRows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        
        for (int i = 1; i <= numRows; i++) {
            h_rowPtrs[i] += h_rowPtrs[i - 1];
        }
        
        CHECK_CUDA(cudaMemcpy(csr.rowPtrs, h_rowPtrs, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_rowPtrs;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 步骤3：复制列索引和值（假设 COO 已按行排序）
    CHECK_CUDA(cudaMemcpy(csr.colIdx, coo.colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(csr.values, coo.values, nnz * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ====================== CPU 参考实现 ======================

/**
 * CPU 密集矩阵-向量乘法（验证用）
 */
void spmv_dense_cpu(int m, int n, const float* A, const float* x, float* y) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}
