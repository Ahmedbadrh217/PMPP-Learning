#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "solution.h"

// ============================================================================
// 常量内存：模板系数
// ============================================================================

__constant__ float coeff[5];

// ============================================================================
// 辅助函数实现
// ============================================================================

void random_data(float* data, int dimx, int dimy, int dimz, float min_val, float max_val) {
    int num_points = dimx * dimy * dimz;
    for (int i = 0; i < num_points; i++) {
        data[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }
}

void store_output(float* output, int dimx, int dimy, int dimz) {
    printf("Output computed for grid %d x %d x %d\n", dimx, dimy, dimz);
    printf("First few output values: ");
    for (int i = 0; i < 5 && i < dimx * dimy * dimz; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
}

void upload_coefficients(float* host_coeff, int num_coeff) {
    // 5 点模板系数（z 方向）
    float stencil_coeff[5] = {-4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    cudaError_t err = cudaMemcpyToSymbol(coeff, stencil_coeff, num_coeff * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to upload coefficients: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ============================================================================
// CUDA 核函数：5 点模板计算
// ============================================================================

__global__ void stencil_kernel(float* output, float* input, int dimx, int dimy, int dimz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= dimx || j >= dimy || k >= dimz) {
        return;
    }
    // 跳过边界点（需要 Halo 数据）
    if (k < 2 || k >= dimz - 2) {
        return;
    }

    int idx = k * dimx * dimy + j * dimx + i;

    // 5 点模板：中心 + z 方向 ±1, ±2
    output[idx] = coeff[0] * input[idx]
                + coeff[1] * input[idx - dimx * dimy]
                + coeff[2] * input[idx + dimx * dimy]
                + coeff[3] * input[idx - dimx * dimy * 2]
                + coeff[4] * input[idx + dimx * dimy * 2];
}

void call_stencil_kernel(float* d_output, float* d_input,
                         int dimx, int dimy, int dimz, cudaStream_t stream) {
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
    dim3 gridSize((dimx + blockSize.x - 1) / blockSize.x,
                  (dimy + blockSize.y - 1) / blockSize.y,
                  (dimz + blockSize.z - 1) / blockSize.z);

    stencil_kernel<<<gridSize, blockSize, 0, stream>>>(d_output, d_input, dimx, dimy, dimz);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int pid;
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        printf("Process %d: Kernel launch failed: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ============================================================================
// 数据服务器进程
// ============================================================================

void data_server(int dimx, int dimy, int dimz, int nreps) {
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    unsigned int num_comp_nodes = np - 1;
    unsigned int first_node = 0;
    unsigned int last_node = np - 2;
    unsigned int num_points = dimx * dimy * dimz;
    unsigned int num_bytes = num_points * sizeof(float);
    
    // 分配输入输出数据
    float* input = (float*)malloc(num_bytes);
    float* output = (float*)malloc(num_bytes);
    
    if (input == NULL || output == NULL) {
        printf("Data server: couldn't allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 初始化输入数据
    random_data(input, dimx, dimy, dimz, 1.0f, 10.0f);
    
    // 计算每个节点的数据量（包含 Halo）
    int slice_per_node = dimz / num_comp_nodes;
    int edge_num_points = dimx * dimy * (slice_per_node + HALO_SIZE);      // 边缘节点
    int int_num_points = dimx * dimy * (slice_per_node + 2 * HALO_SIZE);   // 内部节点
    
    float* send_address = input;
    
    // 发送数据给第一个计算节点
    MPI_Send(send_address, edge_num_points, MPI_FLOAT, first_node, 0, MPI_COMM_WORLD);
    send_address += dimx * dimy * (slice_per_node - HALO_SIZE);
    
    // 发送数据给内部计算节点
    for (int process = 1; process < (int)last_node; process++) {
        MPI_Send(send_address, int_num_points, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
        send_address += dimx * dimy * slice_per_node;
    }
    
    // 发送数据给最后一个计算节点
    if (num_comp_nodes > 1) {
        MPI_Send(send_address, edge_num_points, MPI_FLOAT, last_node, 0, MPI_COMM_WORLD);
    }
    
    // 等待计算完成
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 收集输出数据
    MPI_Status status;
    for (int process = 0; process < (int)num_comp_nodes; process++) {
        int offset = process * slice_per_node * dimx * dimy;
        int recv_points = slice_per_node * dimx * dimy;
        
        MPI_Recv(output + offset, recv_points, MPI_FLOAT, process,
                 DATA_COLLECT, MPI_COMM_WORLD, &status);
    }
    
    // 存储输出
    store_output(output, dimx, dimy, dimz);
    
    // 等待清理完成
    MPI_Barrier(MPI_COMM_WORLD);
    
    free(input);
    free(output);
}

// ============================================================================
// 计算节点进程
// ============================================================================

void compute_node_stencil(int dimx, int dimy, int dimz, int nreps) {
    int np, pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    int server_process = np - 1;
    unsigned int total_z = dimz + 2 * HALO_SIZE;
    unsigned int num_points = dimx * dimy * total_z;
    unsigned int num_bytes = num_points * sizeof(float);
    unsigned int num_halo_points = HALO_SIZE * dimx * dimy;
    unsigned int num_halo_bytes = num_halo_points * sizeof(float);
    
    MPI_Status status;
    
    // 分配主机内存
    float* h_input = (float*)malloc(num_bytes);
    float* h_output = (float*)malloc(num_bytes);
    
    if (h_input == NULL || h_output == NULL) {
        printf("Process %d: failed to allocate host memory\n", pid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 分配设备内存
    float* d_input = NULL;
    float* d_output = NULL;
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_input, num_bytes);
    if (err != cudaSuccess) {
        printf("Process %d: GPU memory allocation failed: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    err = cudaMalloc((void**)&d_output, num_bytes);
    if (err != cudaSuccess) {
        printf("Process %d: GPU output memory allocation failed: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 从数据服务器接收初始数据
    float* rcv_address = h_input + ((pid == 0) ? num_halo_points : 0);
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    
    // 复制到 GPU
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);
    
    // 分配固定内存用于 Halo 交换
    float *h_left_boundary = NULL, *h_right_boundary = NULL;
    float *h_left_halo = NULL, *h_right_halo = NULL;
    
    cudaHostAlloc((void**)&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_right_halo, num_halo_bytes, cudaHostAllocDefault);
    
    // 创建 CUDA 流
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    // 确定邻居进程
    int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 2) ? (pid + 1) : MPI_PROC_NULL;
    
    // 上传模板系数
    float dummy_coeff[5];
    upload_coefficients(dummy_coeff, 5);
    
    // 计算偏移量
    int left_halo_offset = 0;
    int right_halo_offset = dimx * dimy * (HALO_SIZE + dimz);
    int left_stage1_offset = 0;
    int right_stage1_offset = dimx * dimy * (dimz - HALO_SIZE);
    int stage2_offset = num_halo_points;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 迭代计算
    for (int i = 0; i < nreps; i++) {
        // 阶段 1：计算边界区域（stream0）
        call_stencil_kernel(d_output + left_stage1_offset,
                            d_input + left_stage1_offset,
                            dimx, dimy, 3 * HALO_SIZE, stream0);
        call_stencil_kernel(d_output + right_stage1_offset,
                            d_input + right_stage1_offset,
                            dimx, dimy, 3 * HALO_SIZE, stream0);
        
        // 阶段 2：计算内部区域（stream1）- 与通信重叠
        call_stencil_kernel(d_output + stage2_offset,
                            d_input + stage2_offset,
                            dimx, dimy, dimz, stream1);
        
        // 复制边界数据到主机
        cudaMemcpyAsync(h_left_boundary, d_output + num_halo_points,
                        num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_right_boundary, d_output + right_stage1_offset + num_halo_points,
                        num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
        
        cudaStreamSynchronize(stream0);
        
        // MPI Halo 交换
        MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, i,
                     h_right_halo, num_halo_points, MPI_FLOAT, right_neighbor, i,
                     MPI_COMM_WORLD, &status);
        
        MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT, right_neighbor, i,
                     h_left_halo, num_halo_points, MPI_FLOAT, left_neighbor, i,
                     MPI_COMM_WORLD, &status);
        
        // 复制 Halo 数据回 GPU
        cudaMemcpyAsync(d_output + left_halo_offset, h_left_halo,
                        num_halo_bytes, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_output + right_halo_offset, h_right_halo,
                        num_halo_bytes, cudaMemcpyHostToDevice, stream0);
        
        cudaDeviceSynchronize();
        
        // 交换输入输出指针
        float* temp = d_output;
        d_output = d_input;
        d_input = temp;
    }
    
    // 交换回来获取最终结果
    float* temp = d_output;
    d_output = d_input;
    d_input = temp;
    
    // 复制结果到主机并发送给数据服务器
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    
    float* send_address = h_output + num_halo_points;
    MPI_Send(send_address, dimx * dimy * dimz, MPI_FLOAT,
             server_process, DATA_COLLECT, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 释放资源
    free(h_input);
    free(h_output);
    cudaFreeHost(h_left_boundary);
    cudaFreeHost(h_right_boundary);
    cudaFreeHost(h_left_halo);
    cudaFreeHost(h_right_halo);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}
