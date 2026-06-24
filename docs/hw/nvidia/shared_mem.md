# CUDA Shared Memory 与 Block 调度

本文档总结 Nsight Systems（nsys）中 shared memory 相关指标的含义，以及 block / warp / SM 之间的资源与调度关系。示例以 **Jetson Thor（Blackwell GPU）** 上的 TRT/Myelin GEMM kernel 为参照。

---

## 1. nsys 里看到的几个字段

点开单个 CUDA kernel 时，nsys 会显示如下与 shared / local 相关的统计。**这些指标描述的是 kernel launch 的资源配置，不是模型 layer 的 activation 显存占用。**

| 字段 | 粒度 | 含义 |
|------|------|------|
| **Static Shared Memory** | 每 block | 编译期固定的 `__shared__` 大小 |
| **Dynamic Shared Memory** | 每 block | launch 时通过 API / launch 参数申请的 dynamic shared |
| **Shared Memory executed** | 每 block | 该 block 在 SM 上**实际占用**的 shared（含 CUDA 保留） |
| **Local Memory Per Thread** | 每 thread | thread-private local memory（多为 register spill） |
| **Local Memory Total** | 整个 launch | 所有 thread 的 local 总占用（≈ Per Thread × 总线程数） |

### 1.1 Dynamic Shared vs Shared executed

典型 GEMM kernel（CUTLASS / Myelin）：

```text
Dynamic Shared:  232,448 B = 227 KiB    ← launch 时申请的 dynamic shared
Shared executed: 233,472 B = 228 KiB    ← 227 KiB + 1 KiB CUDA 保留
```

多出的 **1,024 B（1 KiB）** 是 CUDA 为每个 block 固定保留的开销，不是异常。

### 1.2 Shared executed 是 per-block，不是全 grid 总和

```text
Grid:  <<<7, 128, 1>>>     → 896 blocks
Block: <<<256, 1, 1>>>     → 256 threads = 8 warps/block
Shared executed: 233,472 B  ← 每个 block 占 228 KiB，不是 896 块相加
```

896 个 block 分 **wave** 调度到各 SM 上，**不会同时**全部驻留。全 grid 若要做总量估算：`896 × 228 KiB`，但任一时刻每个 SM 上只有少量 block 在跑。

### 1.3 Local Memory 与 Shared Memory 的区别

| 类型 | 位置 | 速度 | 典型用途 |
|------|------|------|----------|
| **Shared Memory** | SM 片上 | 很快 | block 内线程协作、GEMM tile 缓冲 |
| **Local Memory** | 逻辑 per-thread，物理多在 global/L1 | 较慢 | register spill、过大的 thread-private 数组 |

nsys 里 **Memory → HtoD/DtoH** 行表示 **数据传输**，不是 shared/local 占用，也不是 layer activation 的 VRAM footprint。

若 **Local Memory Per Thread = 0** 但 **Local Memory Total 很大**，常见于 Myelin 生成 cubin 的元数据不完整，应用 ncu 复核 spill，不宜直接下结论。

---

## 2. SM 级 vs Block 级 shared 上限

以 **Jetson Thor（Blackwell, sm_100 系）** 为例：

| 指标 | 层级 | 值 | 含义 |
|------|------|-----|------|
| 每 SM 最大 shared | SM | **228 KiB** | 一个 SM 上所有 resident block 的 shared **总和**上限 |
| 每 block 最大 dynamic shared | Block | **227 KiB** | 单个 block 可申请的 dynamic shared 上限 |
| CUDA 保留 | Block | **1 KiB** | 每个 block 额外占用，计入 SM 的 228 KiB |

关系：

```text
单 block 顶满 = 227 KiB dynamic + 1 KiB 保留 = 228 KiB = 占满整个 SM shared 池
```

### 2.1 两个限制如何同时起作用

一次 kernel launch 需同时满足：

1. **Block 级**：每个 block 的 dynamic shared ≤ 227 KiB（+ 1 KiB 保留）
2. **SM 级**：同一 SM 上所有 active block 的 shared 总和 ≤ 228 KiB

| 每 block shared executed | 每 SM 最多几个 block |
|--------------------------|---------------------|
| 228 KiB（227+1） | **1** |
| 114 KiB | **2** |
| 57 KiB | **4** |

Occupancy 由 **shared、registers、warps** 等多个约束取最小值决定：

```text
每 SM 最多 block 数 = min(
    floor(228 KiB / 每 block shared),
    floor(65536 regs / 每 block 寄存器),
    floor(64 warps / 每 block warps),
    ...
)
```

示例（Myelin GEMM kernel）：

```text
Shared:   228 KiB / 228 KiB → 1 block/SM   ← 瓶颈
Registers: 65536 / (74×256) → 3 block/SM
Warps:     64 / 8            → 8 block/SM
→ 理论 occupancy ≈ 8/64 = 12.5%（nsys 可能显示 ~16.7%，与 Blackwell 调度模型/元数据有关）
```

---

## 3. Shared memory 不是 Cache，没有「驱逐」

| Cache（L1/L2） | Shared Memory |
|----------------|---------------|
| 硬件自动替换 line | 软件 / kernel 显式分配 |
| 可能被驱逐 | block resident 期间**一直占着** |
| 对 programmer 不透明 | launch 时大小已定 |

**一个 block 的 shared 不会「驱逐」另一个正在执行的 block 的 shared。**

实际发生的是 **排队（wave 调度）**，不是 mid-block 抢占：

```text
SM（228 KiB shared 池）

├── Block 0 正在执行 → 占 228 KiB
│   Block 1~895 在全局队列等待
│
├── Block 0 执行完毕 → 释放 228 KiB
│
└── Block 1 被调度上来 → 再占 228 KiB
```

- **不是**：新 block 来了 → 把旧 block 的 shared 挤掉
- **而是**：旧 block 没跑完 → 新 block **等着**；旧 block 跑完 → shared 释放 → 新 block 才上来

Thor 上 **不支持** 把正在执行的 block 的 shared 中途换出去（无 preemptive shared eviction）。

---

## 4. Block 与 Warp 的调度：谁在「切换」？

**没有「某条 block 指令一执行就切换到另一个 block」的机制。** SM 上的切换发生在 **warp（32 线程）粒度**，由 **硬件 warp scheduler** 决定。

### 4.1 SM 上的结构

```text
一个 SM
├── 最多驻留 N 个 block（受 shared / reg / warp 限制）
│   ├── Block A：8 个 warp
│   ├── Block B：8 个 warp（若资源够）
│   └── ...
│
└── Warp Scheduler（Blackwell 上通常 4 个）
    每周期从「所有 resident block 的所有 ready warp」里选一个执行
```

**切换单位是 warp，不是 block。** 同一 SM 上多个 block 的 warp 交错执行，本质是 **换了一个 warp**。

### 4.2 何时换到「另一个 block 的 warp」？

当 **当前 warp stall（不能继续执行）** 时，scheduler 从 ready 队列选别的 warp，可能来自同一 block，也可能来自 **同一 SM 上另一个 block**：

| 当前 warp 卡住的原因 | 典型场景 |
|---------------------|----------|
| 等 global memory | `load/store` 到 HBM |
| 等 shared 依赖 | shared load 后 pipeline 未 ready |
| Pipeline 空泡 | 指令依赖链未满足 |
| Tensor Core 长操作 | GEMM 多 stage pipeline |

**没有任何 CUDA 指令的语义是「现在切到别的 block」。** 只是当前 warp 等了 → 硬件执行别的 ready warp。

### 4.3 两个 Scheduler，不要混淆

| | **Warp Scheduler** | **Block Scheduler** |
|--|-------------------|---------------------|
| 粒度 | 32 线程（warp） | 整个 block |
| 触发 | warp stall 时 / 每 cycle | **block 全部 warp 执行完毕** |
| 作用 | 隐藏 latency | 决定下一个 block 上哪个 SM |
| 跨 block | 能（同 SM 多个 resident block 时） | block 之间的切换 |

`__syncthreads()` **不会**释放 shared，也 **不会**结束 block；只是 block 内 warp 同步，未到的 warp stall，scheduler 可能去跑同 block 或其他 block 的 ready warp。

### 4.4 大 shared GEMM 的实际行为（每 SM 仅 1 block）

```text
SM0: 只有 Block 0 驻留（228 KiB 占满），8 个 warp

warp0 等 HBM → scheduler 跑 warp1~7（仍是 Block 0 内部切换）
Block 0 跑完  → SM0 释放 → Block 20 上来（下一 wave）
```

因为 **每 SM 只有 1 个 block**，同 SM 上主要是 **同一 block 内 8 个 warp 互相填流水线**，看不到 block 间交错。其余 block 在别的 SM 上跑，或在全局队列等 wave。

---

## 5. 所有 warp 都在等资源时，shared 怎么办？

**Shared memory 一直占着，不会因为在等就被释放或借给别的 block。**

```text
Block 还在 SM 上（resident）  →  shared 一直属于它
Block 所有 warp 都在 stall    →  shared 仍然占着
Block 全部 warp 执行完毕      →  shared 才释放，给下一个 block
```

| 状态 | Warp 在干嘛 | Shared 怎么办 |
|------|-------------|---------------|
| 部分 warp 在算，部分在等 | 正常交错 | 继续占着，被 active warp 读写 |
| **全部 warp 都在等 HBM** | SM 可能空转 | **仍占着**，不能给别的 block |
| Block 全部 warp 结束 | block 生命周期结束 | **释放** |

原因：shared 是 **block 级私有地址空间**，block 内 warp 通过 barrier 协作；若 mid-block 把 shared 让给别的 block，地址空间会乱、数据会被破坏。

### 5.1 与低 Occupancy 的关系

```text
每 block：228 KiB shared，74 regs/thread × 256 threads
每 SM：只能驻留 1 个 block

最坏情况：8 个 warp 同时等 HBM
  → SM 计算单元空闲
  → 228 KiB shared 仍被 Block 0 占着
  → 别的 block 也上不来（装不下）
  → latency 难以被其他 warp/block 隐藏
```

大 tile GEMM 的取舍：**用大 shared 换单次访问效率**，接受 **stall 时难以靠多 block 并行隐藏 latency**。低 occupancy 在此场景下是预期行为，不必视为 bug。

---

## 6. Jetson Thor 硬件参数（参考）

| 项目 | Jetson AGX Thor DevKit（常见值） |
|------|----------------------------------|
| 架构 | Blackwell（`NVIDIA Thor`） |
| Compute Capability | sm_100 系 |
| SM 数量 | 20（TRM 最多 24） |
| 每 SM CUDA Core | 128 → 共 2560 |
| 每 SM 寄存器 | 65536 × 32-bit |
| 每 SM 最大 shared | **228 KiB** |
| 每 block 最大 dynamic shared | **227 KiB** |
| 每 SM 最大 warp | 64 |
| 设备内存 | 128 GB LPDDR5X，~273 GB/s（统一内存） |

确认本机参数：

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
/usr/local/cuda/extras/demo_suite/deviceQuery | egrep "GPU|SM|Shared|Register|Warp"
```

Thor 统一内存下，HtoD/DtoH 与 kernel 共用内存总线；做 kernel 级带宽分析时建议 `trtexec --noDataTransfers` 减少边界拷贝干扰。

---

## 7. 常见误解速查

| 误解 | 正确理解 |
|------|----------|
| Shared executed 是全 grid 所有 block 之和 | **每 block** 的值 |
| executed 比 dynamic 多 1 KiB 是异常 | 正常的 CUDA per-block 保留 |
| SM 有 228 KiB、block 最多 227 KiB，所以 SM 还剩 1 KiB | 一个 block 顶满时 **227+1=228**，占满整个 SM 池 |
| shared 满了会 mid-block 驱逐 | **不会**；block 跑完才释放 |
| 遇到 `__syncthreads` 就切到别的 block | 只是 warp stall；可能跑别的 warp，block 仍 resident |
| occupancy 低 = shared 被浪费所以被踢 | 是 **单 block 太大装不下第二个**，不是驱逐 |
| nsys Memory 行 = layer 显存占用 | 是 **memcpy 流量**，不是 activation footprint |

---

## 8. 与 TRT / nsys 时间线的对应

### 8.1 不同颜色框的层级（嵌套关系）

同一次推理在 nsys 里会被多条轨道、多个抽象层级各画一遍。**时长相同表示嵌套，不是重复计时。**

```text
CPU / TensorRT API 层
  └─ 橙框: ExecutionContext::enqueueV3
         「TRT 运行时调用 enqueue，把这次推理提交到 CUDA stream」

TensorRT 逻辑节点层（NVTX / engine 标注）
  └─ 黄框: ForeignNode[/input_layernorm/.../Mul_2]
         「TRT 图里这块子图的名字（来自 ONNX 路径）；
           ForeignNode 常指交给 Myelin 等后端执行的融合子图」

Myelin 执行引擎层
  └─ 灰框: myelinGraphExecute
         「Myelin 一次性 replay 编译好的子图 / CUDA Graph」

GPU 硬件层
  └─ 蓝框（Kernels）: cutlass gemm、FMHA 等
         「灰框内部实际跑的 CUDA kernel，点开后看 shared/reg/occupancy」
```

| nsys 轨道 | 颜色 | 抽象层级 | 与 shared 的关系 |
|-----------|------|----------|------------------|
| `enqueueV3` | 橙 | TRT API | 提交推理；不直接显示 shared |
| `ForeignNode[...]` / Stack Ranges | 黄 | 逻辑 layer / 子图 | 对应模型语义；不直接显示 shared |
| `myelinGraphExecute` | 灰 | Myelin graph replay | 内部含多个 kernel，各有 per-block shared |
| Kernels on Stream 30 | 蓝 | 实际 CUDA kernel | 点开后看 Dynamic / executed shared、registers、occupancy |
| Stream 29 / 31 | — | HtoD / DtoH | 边界 IO，与 shared 无关 |

**读法**：橙 = TRT 怎么提交；黄 = 提交的是哪块模型子图；灰 = Myelin 怎么执行；蓝 = GPU 真正干的活。竖直对齐 = 同一次 forward 的不同视角。

### 8.2 Shared 与 L1 Cache 的物理关系（Blackwell）

Blackwell（Thor）上 L1 cache 与 shared memory **共用同一块片上存储**（carveout 可在运行时选择比例）。逻辑上：

- **Shared**：block 私有分配，resident 期间不释放
- **L1 Cache**：硬件自动管理，line 可被替换

即使物理存储重叠，**shared 的 block 级语义不变**——不会因为 warp stall 就把 shared carveout 让给别的 block。

### 8.3 Profiling 建议

| 目标 | 建议 |
|------|------|
| 看 per-kernel shared / reg / occupancy | nsys 点蓝框；**关闭 `--useCudaGraph`** |
| 看 layer 耗时 | `trtexec --exportProfile` 或黄框 ForeignNode 名 |
| 看 HBM 带宽 / Tensor Core 利用率 | ncu；**关闭 `--useCudaGraph`** |
| 看线上真实 latency / Graph 行为 | **开启 `--useCudaGraph`**；nsys 看灰框 `myelinGraphExecute` |
| 减少 HtoD/DtoH 干扰 | `trtexec --noDataTransfers`（专注 kernel HBM，非端到端） |

`--useCudaGraph` 会把多次 kernel launch 收成 `cudaGraphLaunch` / `myelinGraphExecute` 长条，timeline 上不易逐 kernel 点开；做 shared/reg 分析时应关 Graph，测 webui 真实延迟时再开。

---

## 9. 一句话总结

- **228 KiB/SM** = SM 硬件 shared 池总容量；**227 KiB/block** = 单 block 可申请 dynamic 上限；差 **1 KiB** 为 CUDA 每 block 固定保留。
- **Shared executed** = **每个 block** 实际占用（227 KiB + 1 KiB），不是全 grid 总和。
- Shared **不是 cache**，block resident 期间 **不会被驱逐或借出**；新 block 要么与已有 block **共享 SM 池**（总和 ≤ 228 KiB），要么 **排队等 wave**。
- SM 上切换发生在 **warp 粒度**；**整个 block 换下** 只在 block **全部 warp 执行完毕** 后发生。
- **所有 warp 都在等资源时，shared 仍占着**——这是大 shared、低 occupancy kernel 在 stall 时难以隐藏 latency 的根本原因。
