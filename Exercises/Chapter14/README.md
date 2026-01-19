# 第十四章：稀疏矩阵计算

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章系统梳理稀疏矩阵存储格式和 SpMV（稀疏矩阵-向量乘法）的 CUDA 实现：

- COO、CSR、ELL、JDS 等存储格式
- 各格式的 SpMV 并行实现
- 格式转换（COO → CSR）
- 混合格式优化（ELL-COO）

**相关博客笔记**：[PMPP-第十四章：稀疏矩阵.md](../../Blogs/PMPP-第十四章：稀疏矩阵.md)

---

## 💻 代码实现

### Exercise01 - SpMV 实现

实现多种稀疏格式的 SpMV kernel。

**代码位置**：`Exercise01/`

**实现列表**：

| 实现 | 格式 | 特点 |
| ---- | ---- | ---- |
| `spmv_coo` | COO | 原子操作累加，最简单 |
| `spmv_csr` | CSR | 每行一线程，最常用 |
| `spmv_ell` | ELL | 列主序，合并访问 |
| `spmv_jds` | JDS | 按行长度排序（练习5） |
| `spmv_hybrid` | ELL-COO | 混合格式（练习4） |
| `coo_to_csr` | 转换 | 直方图+前缀和（练习3） |

**核心代码**：

```cuda
// CSR SpMV - 每行一个线程
__global__ void spmv_csr_kernel(int numRows, const int* rowPtrs, const int* colIdx,
                                 const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        float sum = 0.0f;
        for (int j = rowPtrs[row]; j < rowPtrs[row + 1]; j++) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}
```

#### 运行 Exercise01

```bash
cd Exercise01
make
make run
```

#### 预期输出

```text
================================================================
  第十四章：稀疏矩阵计算
  Sparse Matrix-Vector Multiplication (SpMV)
================================================================

配置:
  矩阵大小: 4 × 4
  非零元素: 8
  稀疏度: 50.0%

=== 正确性验证 ===

1. COO SpMV... ✅ 结果正确！
2. CSR SpMV... ✅ 结果正确！
3. ELL SpMV... ✅ 结果正确！
4. COO to CSR 转换 (练习3)... ✅ 转换正确！
5. JDS SpMV (练习5)... ✅ 结果正确！
6. ELL-COO Hybrid SpMV (练习4)... ✅ 结果正确！
```

---

## 📖 练习题解答

### 练习 1

**题目：** 对于以下稀疏矩阵，分别用 COO、CSR、ELL 和 JDS 格式表示。

![原始矩阵](exercise2.png)

**解答：**

**COO 格式：**

![原始矩阵](exercise2_coo.png)

**CSR 格式：**

![原始矩阵](exercise2_csr.png)

**ELL 格式：**

![原始矩阵](exercise2_ell.png)

**JDS 格式：**

![JDS格式](exercise2_jds.png)

---

### 练习 2

**题目：** 给定 m 行、n 列、z 个非零元素的稀疏矩阵，各格式需要多少整数存储？

**解答：**

| 格式 | 存储空间 | 说明 |
|------|----------|------|
| COO | 3z | rowIdx(z) + colIdx(z) + values(z) |
| CSR | 2z + m + 1 | rowPtrs(m+1) + colIdx(z) + values(z) |
| ELL | 2 × m × K | 需要知道最大行长度 K |
| JDS | 2z + m + K + 1 | 需要知道最大行长度 K |

注意：ELL 和 JDS 需要额外信息（最大行长度 K）才能精确计算。

---

### 练习 3

**题目：** 使用并行计算原语（直方图、前缀和）实现 COO 到 CSR 的转换。

**解答：**

```cuda
// 步骤1：并行直方图 - 统计每行元素数
__global__ void computeHistogram(int nnz, int* rowIdx, int* rowPtrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&rowPtrs[rowIdx[i] + 1], 1);
    }
}

// 步骤2：前缀和 - 计算行指针
// 使用 thrust::exclusive_scan 或自定义扫描

// 步骤3：复制列索引和值（假设 COO 已按行排序）
```

完整实现见 `solution.cu` 中的 `coo_to_csr` 函数。

---

### 练习 4

**题目：** 实现 ELL-COO 混合格式的 SpMV，ELL 在 GPU 执行，COO 溢出部分在 CPU 执行。

**解答：**

```cuda
void spmv_hybrid(const ELLMatrix& ellPart, const COOMatrix& cooPart, 
                 const float* d_x, float* d_y) {
    // 1. GPU 执行 ELL 部分
    spmv_ell_kernel<<<...>>>(ellPart, d_x, d_y);
    
    // 2. 拷贝部分结果到 CPU
    cudaMemcpy(h_y, d_y, ...);
    cudaMemcpy(h_x, d_x, ...);
    
    // 3. CPU 执行 COO 溢出部分
    for (int i = 0; i < cooPart.nnz; i++) {
        h_y[cooPart.rowIdx[i]] += cooPart.values[i] * h_x[cooPart.colIdx[i]];
    }
    
    // 4. 拷贝回 GPU
    cudaMemcpy(d_y, h_y, ...);
}
```

完整实现见 `solution.cu` 中的 `spmv_hybrid` 函数。

---

### 练习 5

**题目：** 实现 JDS 格式的并行 SpMV kernel。

**解答：**

```cuda
__global__ void spmv_jds_kernel(int numRows, int numTiles, const int* colIdx,
                                 const float* values, const int* rowPerm, 
                                 const int* iterPtr, const float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    
    float sum = 0.0f;
    for (int t = 0; t < numTiles; t++) {
        int idx = iterPtr[t] + tid;
        if (idx < iterPtr[t + 1]) {
            sum += values[idx] * x[colIdx[idx]];
        }
    }
    // 按原始行顺序写回
    y[rowPerm[tid]] = sum;
}
```

完整实现见 `solution.cu` 中的 `spmv_jds` 函数。

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 💡 学习建议

1. **理解格式特点**：COO 简单、CSR 通用、ELL 规则、JDS 平衡
2. **权衡空间和性能**：ELL 可能浪费空间但访问规则
3. **考虑负载均衡**：行长度差异大时需要特殊处理
4. **生产环境用库**：cuSPARSE 提供高度优化的实现

## 🚀 下一步

完成本章学习后，继续学习：

- 第十五章：图遍历
- 第十六章：深度学习
- 第十七章：迭代磁共振成像重建

---

**学习愉快！** 🎓
