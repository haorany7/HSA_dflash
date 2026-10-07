# DFlash GPU + NPU 异构方案 (Qwen3-8B, Ryzen AI 9 HX 370)

> 状态：进行中（2026-10-07）。本文根据两次 Claude Code 会话的日志和当前工作区代码整理，
> 目的是保存进度。数字均来自同一个 prompt（"How many positive whole-number divisors does
> 196 have? Explain briefly."），greedy，128 个输出 token，除非另有说明。

## 1. 目标

在这台 AMD APU 上跑 DFlash 推测解码，支持 `Qwen/Qwen3-8B`：

- **Target**（`Qwen/Qwen3-8B`，bf16 16 GB）→ **iGPU**（Radeon 890M，ROCm），W8A8 量化
- **Draft**（`z-lab/Qwen3-8B-DFlash-b16`，约 1B，bf16 2 GB，block=16，
  `target_layer_ids=[1, 9, 17, 25, 33]`）→ **NPU**（Strix，aie2p，mlir-aie whole_array GEMM）
- CPU 只负责胶水代码

## 2. 硬件与约束

| 项 | 值 |
|---|---|
| CPU | Ryzen AI 9 HX 370，12 线程 torch |
| 内存 | 23 GB LPDDR5X，CPU/GPU/NPU 共享（峰值带宽约 120 GB/s） |
| GPU | Radeon 890M，gfx1150，torch 2.13.0+rocm7.2，bf16 GEMM 实测 8.8 TFLOPS |
| GPU 可用显存 | **11.5 GiB**（GTT 上限）。bf16 target 16.4 GB 放不下 → 必须量化（或 root 改 `amdgpu.gttsize` 后重启） |
| NPU | `/dev/accel/accel0`，XRT 2.26.0，aie2p，按 8 列使用 |

## 3. 实测结果

### 3.1 基线（全 CPU）

| 配置 | decode | 平均接受长度 |
|---|---|---|
| 自回归，CPU | 3.17 tok/s | 1.00 |
| DFlash，target + draft 都在 CPU | 12.4–14.0 tok/s | 6.05 |

DFlash greedy 输出和自回归的前 48 个 token 完全一致，第 48 个 token 起分叉，两边都是合理的续写。
原因是 bf16 舍入：16 个 token 一起验证和逐个 token 验证的舍入不同，不是 DFlash 的 bug。

### 3.2 Profile（target 在 CPU）

| Draft 设备 | target 前向/步 | draft body/步 | draft lm_head/步 | decode |
|---|---|---|---|---|
| CPU | 364 ms (**83%**) | 51.5 ms | 22.6 ms | 13.98 tok/s |
| NPU（仅 Linear） | 383 ms | **376 ms** | 24.3 ms | 7.78 tok/s |

结论：瓶颈在 target，所以先把 target 搬上 GPU。现在的 NPU draft 比 CPU 慢约 7 倍。

### 3.3 Target + draft 都在 GPU（W8A8）

| 配置 | target/步 | draft body + lm_head | 接受长度 | decode | GPU 峰值显存 |
|---|---|---|---|---|---|
| W8A8，bf16 lm_head | 203 ms | 30 + 17 ms | 7.94 | 32.1 tok/s | 10.34 GiB |
| + int8 lm_head | 192 ms | 29 + 10 ms | 7.06 | 30.8 tok/s | 9.75 GiB |
| **+ int8 lm_head + `torch.compile`** | **164 ms** | **25 + 10 ms** | 7.06 | **35.8 tok/s** | 9.75 GiB |

- 相比 CPU target（14.0 tok/s）快约 2.6 倍。
- int8 lm_head 会让措辞略有变化，接受长度从 7.94 降到 7.06。目前**只测了一个 prompt**，
  不能确定这个下降是否真实存在。
- 单个 int8 GEMM（`torch._int_mm`）实测约 60 GB/s，内存带宽上限约 120 GB/s，还有余量。
  M=16 时各 shape 耗时（eager / 只算 int_mm / compiled，单位 ms）：
  4096×4096 0.47/0.34/0.37，4096×1024 0.21/0.12/0.12，
  4096×12288 1.03/0.84/0.86，12288×4096 1.12/0.91/0.94。

### 3.4 NPU 单层 GEMM（`NpuLinear`，rows=16，修复后）

| K × N | 相对误差 | NPU 稳态 | CPU |
|---|---|---|---|
| 4096 × 4096 | 6.8e-5 | 4.0 ms | 1.3 ms |
| 4096 × 1024 | 5.9e-5 | 1.5 ms | 0.3 ms |
| 4096 × 12288 | 4.3e-5 | 10.4 ms | 2.1 ms |
| 12288 × 4096（down_proj，按 K 切分） | 7.2e-5 | 11.7 ms | 2.2 ms |
| 20480 × 4096（fc，按 K 切分） | 6.4e-5 | 22.0 ms | 5.7 ms |
| 4096 × 4096，rows=100 | 3.9e-5 | 6.0 ms | 3.0 ms |

每个 shape 首次 JIT 编译约 5 s，之后走 `~/.npu/cache`。

## 4. 异构架构

```
              ┌──────────── GPU (890M, W8A8) ─────────────┐
 prompt ──►   │ target prefill / 16-token verify           │
              │   → hidden states of layers 1,9,17,25,33   │──┐ context feature
              │ target lm_head（draft 共用）                │  │ (≈16 × 5×4096 bf16 / step)
              └───────────────▲────────────────────────────┘  │
                              │ draft hidden → logits          ▼
              ┌───────────────┴──── NPU (aie2p, 8 cols) ──────────────┐
              │ draft body：fc / q,k,v,o / gate,up,down（bf16 GEMM,     │
              │ f32 输出）；norm / RoPE / attention 目前仍在 torch 侧     │
              └───────────────────────────────────────────────────────┘
```

- 每一步都有串行依赖：draft 需要 target 上一步验证后的 hidden states，target 又需要 draft
  的提议才能验证。所以**不改算法的话，GPU 和 NPU 不能同时工作**。
- 在 GPU 方案里 draft 只占每步约 35/200 ms（<20%）。就算 NPU 把 draft 的时间降到 0，
  端到端最多也只快 15–20%。而 NPU draft 先要从 376 ms 降到 25 ms 以下才能打平。
- 把 draft 放到 NPU 的其他好处：GPU 能腾出约 1 GB 显存（给更长的 KV cache 或 bf16 lm_head），
  并为以后的流水化留出空间，例如让 draft 预先起草下一块，验证通过就直接用（需要改算法）。
- 三个设备共用同一块 LPDDR5X，所以 GPU 和 NPU 之间原则上可以零拷贝。
  目前的实现是 GPU → CPU numpy → NPU buffer，每层都要拷一次。

## 5. 代码现状

### 5.1 `/notebooks/dflash` 仓库（尚未 commit）

| 文件 | 改动 |
|---|---|
| `dflash/npu.py`（新增） | `NpuLinear` / `offload_linears`：把 draft 中所有能切块的 `nn.Linear` 换成 mlir-aie `whole_array` bf16 GEMM（`@iron.jit`）。tile 为 m,k,n=16,64,64，8 列，`b_col_maj=1`（直接用 nn.Linear 的权重布局）。已修两个问题：① 输出改为 **f32**（bf16 输出在 K=4096 时误差 0.11）；② K>4096 时**沿 K 切成多块**，在 host 上把部分和加起来（否则会触发 DMA stride 超过 2^20 的编译错误）。 |
| `dflash/cli.py` | 新增 `--device`（默认 `cuda:0`）和 `--draft-device {same,npu}` |
| `dflash/benchmark.py` | `load_transformers_models(..., draft_device)`。传 `npu` 时调用 `offload_linears(draft)` |
| `dflash/model.py` | `_cuda_time()` 在没有 GPU 时跳过 synchronize |

### 5.2 `scripts/hetero/`（从会话日志还原）

原来放在 `/tmp` scratchpad，已被清掉。本次通过回放日志里的 Write 和 patch 还原到仓库，均能通过 `py_compile`，但**还原后没有重新运行过**。

| 文件 | 作用 |
|---|---|
| `gpu_w8a8.py` | `W8A8Linear`：权重 int8，按输出通道对称量化；激活 int8，按 token 动态量化；用 `torch._int_mm` 计算，M 至少补到 32 且对齐到 8。`load_target_w8a8` / `load_draft_w8a8` 先在 meta 上建空模型，再逐个 tensor 流式量化到 GPU（避免 16 GB 的 bf16 整体进内存；早先因为 buffers 字典持有 GPU 占位张量而 OOM，已修）。lm_head 按 8192 行分块量化。环境变量 `W8A8_COMPILE=1` 打开 `torch.compile(dynamic=True)`。 |
| `profile_dflash.py` | 分阶段计时：target_fwd、target lm_head、draft_body、draft_lm_head、other。参数 `--device`、`--draft-device`、`--w8a8`；环境变量 `W8A8_HEAD=1` 把 lm_head 也量化成 int8。会先用相同长度跑一遍预热。 |
| `run_dflash.py` | 冒烟测试：DFlash 与自回归（`block_size=1`）对比，报告第一个分叉的 token |
| `test_npu_linear.py` | 测 `NpuLinear` 各 shape 的精度和延迟，与 CPU 对比 |

### 5.3 还没有的部分（"改了一半"）

- **target 用 GPU W8A8 + draft 用 NPU 的组合路径还没打通。** `npu.py` 的输出会经过
  `.to(x.device)`，理论上能回到 GPU，但还没测过。`load_draft_w8a8` 和 `offload_linears` 是两条独立的路径。
- `gpu_w8a8.py` 还没有并入 `dflash/` 包或 CLI（例如 `--target-quant w8a8`）。
- 质量只验证过 1 个 prompt。

## 6. 运行环境

```bash
cd /notebooks/dflash
source /opt/xilinx/xrt/setup.sh
export PYTHONPATH=/home/mliraie/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/python:/opt/xilinx/xrt/python
export PATH=/home/mliraie/mlir-aie/ironenv/bin:$PATH   # aiecc 等工具
# MLIR_AIE_SRC 默认是 /notebooks/mlir-aie（whole_array 设计所在的位置）

# GPU W8A8（当前最好配置，35.8 tok/s）
W8A8_COMPILE=1 W8A8_HEAD=1 PYTHONPATH=scripts/hetero:$PYTHONPATH \
  .venv/bin/python scripts/hetero/profile_dflash.py --device cuda --w8a8

# CPU target + NPU draft
PYTHONPATH=scripts/hetero:$PYTHONPATH \
  .venv/bin/python scripts/hetero/profile_dflash.py --device cpu --draft-device npu
```

注意事项：
- dflash 的 `.venv`（torch-ROCm + transformers）里额外装了 `ml_dtypes aiofiles rich cloudpickle`，这样 IRON 才能 import。
- 找不到 `pyxrt` 时，IRON 会**静默**退回到只有 CPU 的 tensor 类，然后报 `Unsupported device: npu`。
  所以一定要先 source XRT，并把 `/opt/xilinx/xrt/python` 加进 PYTHONPATH。
- checkpoint 在 `~/.cache/huggingface/hub/`，两个都已下载完整。可以设 `HF_HUB_OFFLINE=1`。

## 7. 下一步（待定）

上次会话最后给了三个选项，用户还没选：

1. **NPU draft 优化（第 3 步）**，目标是 draft body 从 376 ms 降到 25 ms 以下：
   - 预分配并复用 A/C buffer，不再每次 `iron.tensor` / `iron.zeros`
   - 去掉 torch→numpy→NPU 的来回拷贝，利用共享内存做零拷贝
   - 行数补齐太多：目前 `_M_ALIGN=128`，而 draft 只有 16–32 行。可以改用 m 更小的 tile，或专门为 M=16/32 写设计
   - 合并 q/k/v、gate/up，减少调用次数（目前每步约 36 次）
   - 让激活在层与层之间留在 NPU 上；长期可以把整层放到 NPU 上（融合 norm、SiLU）
   - K 切分的部分和改在 NPU 上累加
2. **继续优化 GPU 上的 target**：合并 qkv、gate/up，写 Triton int8 kernel，目标是接近 120 GB/s
3. **先把代码并入 dflash 包，并用多个 prompt 验证质量**（W8A8 和 int8 lm_head 对接受长度的影响）

先按原路线排序：1）profile ✅ → 2）target 上 GPU ✅ → 3）NPU draft + 组合异构 ⏳
