# 第十七章：迭代式磁共振成像重建

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍 MRI 图像重建的 GPU 加速技术：

- 非笛卡尔采样与非均匀 FFT（NUFFT）
- 迭代重建与共轭梯度法（CG）
- F^H D 核心计算的 GPU 优化
- 循环优化技术：循环分裂、循环交换

**相关博客笔记**：[PMPP-第十七章：迭代式磁共振成像重建.md](../../Blogs/PMPP-第十七章：迭代式磁共振成像重建.md)

---

## 💻 代码实现

### Exercise01 - 共轭梯度法 (CG)

实现求解对称正定线性系统 Ax = b 的共轭梯度法。

**代码位置**：`Exercise01/`

| 实现 | 特点 |
| ---- | ---- |
| `cg_solve_cpu` | CPU 串行版本 |
| `cg_solve_gpu` | GPU 并行版本（向量操作并行） |
| `vector_dot/axpy/xpay` | 基础向量操作 |
| `matvec_multiply` | 矩阵-向量乘法 |

```bash
cd Exercise01 && make && make run
```

---

### Exercise02 - F^H D 核心计算

实现 MRI 重建的核心运算：傅里叶变换的共轭转置乘以数据。

**代码位置**：`Exercise02/`

| 实现 | 特点 |
| ---- | ---- |
| `fhd_compute_cpu` | CPU 参考实现 |
| `fhd_compute_gpu` | GPU 基础版本 |
| `fhd_compute_gpu_optimized` | 使用寄存器优化 |
| `compute_mu_gpu` | 循环分裂：预计算 Mu |

```bash
cd Exercise02 && make && make run
```

---

### Exercise03 - NUFFT Gridding

实现非均匀 FFT 的网格化操作，用于非笛卡尔采样轨迹。

**代码位置**：`Exercise03/`

| 实现 | 特点 |
| ---- | ---- |
| `gridding_cpu/gpu` | 非均匀→规则网格 |
| `degridding_cpu/gpu` | 规则网格→非均匀 |
| `kaiser_bessel` | Kaiser-Bessel 插值核 |
| `density_compensation` | 采样密度补偿 |

```bash
cd Exercise03 && make && make run
```

---

## 📖 练习题解答

### 练习 1：循环分裂 (Loop Fission)

**题目**：对 Figure 17.4 中的 F^H D 代码进行循环分裂分析。

**原始代码**：

```cpp
for (int m = 0; m < M; m++) {
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
    for (int n = 0; n < N; n++) {
        float expFhD = 2*PI*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
        rFhD[n] += rMu[m]*cos(expFhD) - iMu[m]*sin(expFhD);
        iFhD[n] += iMu[m]*cos(expFhD) + rMu[m]*sin(expFhD);
    }
}
```

**(a) 分裂前执行顺序：**

- m=0: 计算 rMu[0], iMu[0] → 内循环 n=0..N-1
- m=1: 计算 rMu[1], iMu[1] → 内循环 n=0..N-1
- ...依此类推

**(b) 分裂后执行顺序：**

```cpp
// 第一个循环：预计算所有 Mu
for (int m = 0; m < M; m++) {
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
}
// 第二个循环：使用预计算的 Mu
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) { ... }
}
```

**(c) 结果是否相同：** ✅ 相同

- rMu/iMu 在内循环使用前已全部计算完成
- 累加操作顺序不影响最终结果

---

### 练习 2：循环交换 (Loop Interchange)

**题目**：分析 Figure 17.9 中循环交换的影响。

**(a) 交换前（外 n 内 m）：**

```
(n=0, m=0), (n=0, m=1), ..., (n=0, m=M-1)
(n=1, m=0), (n=1, m=1), ..., (n=1, m=M-1)
...
```

**(b) 交换后（外 m 内 n）：**

```
(m=0, n=0), (m=0, n=1), ..., (m=0, n=N-1)
(m=1, n=0), (m=1, n=1), ..., (m=1, n=N-1)
...
```

**(c) 结果是否相同：** ✅ 相同

- 局部变量 expFhD, cArg, sArg 每次迭代重新计算
- 累加操作满足交换律，顺序无影响

---

### 练习 3：寄存器使用分析

**题目**：分析 Figure 17.11 中 x[] 与 kx[] 访问模式的差异。

| 数组 | 索引 | 访问次数 | 寄存器优化 |
|------|------|----------|-----------|
| x[n], y[n], z[n] | n（线程ID） | 每线程 1 次 | ✅ 适合加载到寄存器 |
| kx[m], ky[m], kz[m] | m（循环变量） | 每线程 M 次 | ❌ 不适合 |

**原因**：

- `n` 对每个线程是常量，`x[n]` 在整个循环中不变
- `m` 是循环变量，`kx[m]` 每次迭代都不同
- 将 `kx[m]` 加载到寄存器会浪费 M 个寄存器，每线程只用一次

---

## 📁 项目结构

```
Exercise01/         # 共轭梯度法 (CG)
Exercise02/         # F^H D 核心计算
Exercise03/         # NUFFT Gridding
lenna.png           # 测试图像
fft_representation*.png  # FFT 可视化
```

---

## 🔧 开发环境

- CUDA Toolkit 11.0+

---

## 📚 参考资料

- PMPP 第四版 Chapter 17
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-17)
- [PMPP-第十七章：迭代式磁共振成像重建.md](../../Blogs/PMPP-第十七章：迭代式磁共振成像重建.md)
