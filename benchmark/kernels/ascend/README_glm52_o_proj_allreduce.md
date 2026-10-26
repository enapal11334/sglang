# GLM-5.2 TP16 o_proj + AllReduce 流水采集

目标：对齐提供的服务流水，使用 M16、TP16、静态 W8A8、NZ 权重、无 INT32
bias，矩乘输出 BF16 `[16,6144]`，随后 AllReduce count=98304（192 KiB）。
数据内部构造，不加载模型。

## 执行

在 Ascend Linux 服务器的仓库根目录执行下面这一段命令。环境需安装匹配的
PyTorch、torch_npu 和 CANN，并能访问 16 个逻辑 NPU 设备。
脚本和同目录的 `glm52_o_proj_module.py` 必须同时存在，无需安装 SGLang。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export STREAMS_PER_DEVICE=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=650
export TASK_QUEUE_ENABLE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ASCEND_LAUNCH_BLOCKING

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=16 \
  benchmark/kernels/ascend/bench_glm52_o_proj_allreduce.py \
  --tp-size 16 --tokens 16 \
  --hidden-size 6144 --num-heads 64 --v-head-dim 256 \
  --quant static --weight-format nz \
  --graph \
  --warmup 20 --iterations 50 \
  --profile-dir profiles/glm52_aligned \
  --output-dir results/glm52_aligned
```

HCCL buffer 和 task queue 采用提供的 decode 启动配置；若所比较服务进程的
实际环境不同，应使用该次采集的值。SGLANG 环境变量不控制本用例。
默认使用 eager Module，不调用 `torch.compile`、不依赖 TorchAir。
已移除 `--compile-mode` 和 `--no-compile` 参数。权重转换、精度校验和预热
均在正式计时之外。上述命令使用手工 NPU Graph，保持 NZ 权重并减少下发开销。

默认在导入 Torch、初始化 NPU 前按 LOCAL_RANK 自动绑核，只使用当前 cpuset
允许的 CPU；同一物理核的可用 SMT 线程分配给同一 rank，rank 间不共享核组。
这不是 NPU 拓扑感知的 NUMA/内存绑定；已有外部绑核时使用 `--no-cpu-affinity`。
核组数不足 16 会报错，应扩充容器 cpuset 或通过外部工具安排绑定。
每 rank 打印 CPU 列表，结果 JSON 也保存 cpu_affinity。

采集器启动后执行 3 次准备迭代，再执行一次设备同步、跨 rank barrier 和设备同步，
然后才开始正式计时。该阶段在 Timeline 中标记为 `benchmark_setup`，包含准备
计算和通信，分析时请排除；正式迭代仍为指定的 50 次。
每轮不额外添加 barrier，不会把实际通信等待人为搬到另一项同步中。
这些措施降低初始化和调度扰动，不能保证所有 rank 同时到达或消除全部等待。

## 查看与核对

在 MindStudio Insight 中导入 `profiles/glm52_aligned/rank_<rank>/` 下的
`*_ascend_pt` 目录，打开 Timeline。时间线通常位于其中的
`ASCEND_PROFILER_OUTPUT/trace_view.json`。
`results/glm52_aligned/rank_<rank>.json` 仅为尺寸和耗时汇总。

| 核对项 | 目标 |
| --- | --- |
| 量化后激活 | INT8 `[16,1024]`，ND |
| 右矩阵 | INT8，FRACTAL_NZ；逻辑形状 `[1024,6144]` |
| 反量化 scale | FP32 `[6144]`，ND |
| INT32 bias | 不传入 |
| 矩乘输出 | BF16 `[16,6144]`，ND |
| 通信 | 16-rank AllReduce SUM，BF16，count=98304 |
| 最终输出 | 每 rank BF16 `[16,6144]` |

服务右矩阵的 `[192,64,16,32]` 是 NZ 物理存储形状。
分配权重前设置 `allow_internal_format=True`，完成 `npu_format_cast` 后
通过 `get_npu_format` 验证实际格式为 29；Module 构造后、预热后和保存时再次检查。
格式不符合要求直接报错，不静默采集 ND。rank0 打印实际格式，JSON 的
`weight_actual_npu_format` 保存编号。仍以新流水中 MatMul 的 FRACTAL_NZ 输入
确认设备实际执行格式；检查和格式转换均不放入计时循环。

本命令对齐所贴流水的计算和通信结构。服务执行模式、CPU 绑核、各 rank
到达通信的时间差仍可能影响耗时，不保证相同微秒数；尚未在 910C 实机验证。

## 代码入口

- `glm52_o_proj_module.py`：Module 保存量化参数，forward 执行投影与 AllReduce。
- `bench_glm52_o_proj_allreduce.py`：build_module 构造 eager Module；run 串联
  数据构造、校验、预热、采集和保存。
- 后续融合实现可通过现有 impl 入口接入，签名为
  `fn(x, params, process_group)`，返回完整归约后的 BF16 `[M,6144]`。
