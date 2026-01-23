// ============================================================================
// solution.cu - 第十七章练习1: 共轭梯度法 (Conjugate Gradient)
// ============================================================================
// 求解对称正定线性系统 Ax = b
// 对应参考仓库 cg_algo.py 的 C++/CUDA 实现
// ============================================================================

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cmath>
#include <cstring>
#include <cstdio>

// ============================================================================
// CUDA Kernel 定义
// ============================================================================

#define BLOCK_SIZE 256

/**
 * @brief 向量内积归约 kernel
 */
__global__ void dot_kernel(const float* x, const float* y, float* partial_sums, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算一个乘积
    float sum = 0.0f;
    while (idx < n) {
        sum += x[idx] * y[idx];
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief AXPY: y = alpha * x + y
 */
__global__ void axpy_kernel(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

/**
 * @brief XPAY: y = x + beta * y
 */
__global__ void xpay_kernel(const float* x, float beta, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + beta * y[idx];
    }
}

/**
 * @brief 矩阵-向量乘法: y = A * x
 */
__global__ void matvec_kernel(const float* A, const float* x, float* y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col];
        }
        y[row] = sum;
    }
}

/**
 * @brief 向量复制: y = x
 */
__global__ void copy_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// ============================================================================
// CPU 实现
// ============================================================================

float vector_dot_cpu(const float* x, const float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

void vector_axpy_cpu(float alpha, const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}

void vector_copy_cpu(const float* x, float* y, int n) {
    memcpy(y, x, n * sizeof(float));
}

void vector_xpay_cpu(const float* x, float beta, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + beta * y[i];
    }
}

void matvec_multiply_cpu(const float* A, const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

float vector_norm_cpu(const float* x, int n) {
    return sqrtf(vector_dot_cpu(x, x, n));
}

int cg_solve_cpu(const float* A, const float* b, float* x,
                 int n, float tol, int max_iter) {
    // 分配临时向量
    float* r = new float[n];   // 残差
    float* p = new float[n];   // 搜索方向
    float* Ap = new float[n];  // A * p
    
    // r = b - A*x
    matvec_multiply_cpu(A, x, Ap, n);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
    }
    
    // p = r
    vector_copy_cpu(r, p, n);
    
    float rr = vector_dot_cpu(r, r, n);
    float r_norm = sqrtf(rr);
    
    int k;
    for (k = 0; k < max_iter && r_norm > tol; k++) {
        // Ap = A * p
        matvec_multiply_cpu(A, p, Ap, n);
        
        // alpha = (r · r) / (p · Ap)
        float pAp = vector_dot_cpu(p, Ap, n);
        float alpha = rr / pAp;
        
        // x = x + alpha * p
        vector_axpy_cpu(alpha, p, x, n);
        
        // r = r - alpha * Ap
        vector_axpy_cpu(-alpha, Ap, r, n);
        
        // 新残差范数
        float rr_new = vector_dot_cpu(r, r, n);
        r_norm = sqrtf(rr_new);
        
        // beta = rr_new / rr_old
        float beta = rr_new / rr;
        
        // p = r + beta * p
        vector_xpay_cpu(r, beta, p, n);
        
        rr = rr_new;
    }
    
    delete[] r;
    delete[] p;
    delete[] Ap;
    
    return k;
}

// ============================================================================
// GPU 实现
// ============================================================================

// 静态设备内存用于归约
static float* d_partial_sums = nullptr;
static float* h_partial_sums = nullptr;
static int max_blocks = 0;

static void ensure_reduction_buffers(int num_blocks) {
    if (num_blocks > max_blocks) {
        if (d_partial_sums) cudaFree(d_partial_sums);
        if (h_partial_sums) delete[] h_partial_sums;
        
        max_blocks = num_blocks;
        cudaMalloc(&d_partial_sums, max_blocks * sizeof(float));
        h_partial_sums = new float[max_blocks];
    }
}

float vector_dot_gpu(const float* d_x, const float* d_y, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 1024);  // 限制块数
    
    ensure_reduction_buffers(num_blocks);
    
    dot_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, d_y, d_partial_sums, n);
    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        sum += h_partial_sums[i];
    }
    return sum;
}

void vector_axpy_gpu(float alpha, const float* d_x, float* d_y, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpy_kernel<<<num_blocks, BLOCK_SIZE>>>(alpha, d_x, d_y, n);
}

void vector_copy_gpu(const float* d_x, float* d_y, int n) {
    cudaMemcpy(d_y, d_x, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

void vector_xpay_gpu(const float* d_x, float beta, float* d_y, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    xpay_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, beta, d_y, n);
}

void matvec_multiply_gpu(const float* d_A, const float* d_x, float* d_y, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_kernel<<<num_blocks, BLOCK_SIZE>>>(d_A, d_x, d_y, n);
}

float vector_norm_gpu(const float* d_x, int n) {
    return sqrtf(vector_dot_gpu(d_x, d_x, n));
}

int cg_solve_gpu(const float* d_A, const float* d_b, float* d_x,
                 int n, float tol, int max_iter) {
    // 分配设备临时向量
    float *d_r, *d_p, *d_Ap;
    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_Ap, n * sizeof(float));
    
    // r = b - A*x
    matvec_multiply_gpu(d_A, d_x, d_Ap, n);
    
    // r = b - Ap (使用 copy + axpy)
    vector_copy_gpu(d_b, d_r, n);
    vector_axpy_gpu(-1.0f, d_Ap, d_r, n);
    
    // p = r
    vector_copy_gpu(d_r, d_p, n);
    
    float rr = vector_dot_gpu(d_r, d_r, n);
    float r_norm = sqrtf(rr);
    
    int k;
    for (k = 0; k < max_iter && r_norm > tol; k++) {
        // Ap = A * p
        matvec_multiply_gpu(d_A, d_p, d_Ap, n);
        
        // alpha = (r · r) / (p · Ap)
        float pAp = vector_dot_gpu(d_p, d_Ap, n);
        float alpha = rr / pAp;
        
        // x = x + alpha * p
        vector_axpy_gpu(alpha, d_p, d_x, n);
        
        // r = r - alpha * Ap
        vector_axpy_gpu(-alpha, d_Ap, d_r, n);
        
        // 新残差范数
        float rr_new = vector_dot_gpu(d_r, d_r, n);
        r_norm = sqrtf(rr_new);
        
        // beta = rr_new / rr_old
        float beta = rr_new / rr;
        
        // p = r + beta * p
        vector_xpay_gpu(d_r, beta, d_p, n);
        
        rr = rr_new;
    }
    
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    
    return k;
}
