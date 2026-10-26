"""Synthetic GLM-5.2 W8A8 o_proj + HCCL all-reduce, without SGLang imports.

No checkpoint, attention, KV cache, model runner, or decode is used.
See README_glm52_o_proj_allreduce.md for launch and profiling commands.

张量记号：M=tokens，K=num_heads*v_head_dim/tp_size（本 rank 输入宽度），
N=hidden_size。GLM-5.2 TP16 默认 K=1024、N=6144；所有 NPU 张量位于
本进程绑定的 npu:LOCAL_RANK。NZ 表示物理存储格式，文中的 shape 为逻辑形状。
"""

import argparse
import importlib
import json
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    """解析命令行并检查尺寸和迭代次数。

    输入：无显式参数，从 sys.argv 读取命令行。
    输出：argparse.Namespace，字段含义如下：
        tp_size(int)：TP 进程数；tokens(int)：M；hidden_size(int)：N；
        num_heads/v_head_dim(int)：全局 attention heads 数及每 head 的 V 维度。
        quant(str)：static/dynamic；weight_format(str)：nz/nd。
        quant_bias(bool)：是否传零偏置。
        warmup/iterations(int)：预热/测量次数，均须为正整数。
        graph(bool)：是否 capture/replay；profile_dir(str|None)：采集目录；
        no_cpu_affinity(bool)：保留外部 CPU 绑定，默认按本机 rank 自动绑核。
        output_dir(str)：结果目录；impl(str|None)：module:function 替换入口；
        describe(bool)：仅打印尺寸，不初始化 NPU。
    异常：参数无效时 argparse 打印原因并以 SystemExit 退出。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp-size", type=int, default=16)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=6144)
    p.add_argument("--num-heads", type=int, default=64)
    p.add_argument("--v-head-dim", type=int, default=256)
    p.add_argument("--quant", choices=["static", "dynamic"], default="static")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--no-cpu-affinity", action="store_true", help="Keep externally configured CPU affinity")
    p.add_argument("--profile-dir", help="torch_npu profiler output, one dir per rank")
    p.add_argument("--output-dir", default="o_proj_results")
    p.add_argument("--weight-format", choices=["nz", "nd"], default="nz")
    p.add_argument("--quant-bias", action="store_true", help="Pass zero INT32 bias on rank0")
    p.add_argument("--impl", help="module:function implementing fn(x, params, process_group)")
    p.add_argument("--describe", action="store_true", help="Print shapes without NPU imports")
    args = p.parse_args()
    if min(args.tp_size, args.tokens, args.hidden_size, args.num_heads,
           args.v_head_dim, args.warmup, args.iterations) < 1:
        p.error("sizes and iteration counts must be positive")
    if args.num_heads % args.tp_size:
        p.error("num-heads must be divisible by tp-size")
    return args


def o_proj(x, params):
    """执行本 rank 的激活量化与 W8A8 输出投影，不做跨 rank 归约。

    输入：
        x：NPU BF16 Tensor [M,K]，已按 TP 切分的 attention 输出。
        params：SimpleNamespace，所有 tensor 与 x 同设备：
            quant(str)：static 或 dynamic。
            weight：INT8 [K,N]，NZ 或 ND 存储。
            input_scale_reciprocal：BF16 [K]，静态激活量化 scale 的倒数。
            input_offset：BF16 [K]，静态激活量化 offset。
            deq_scale：FP32 [N]，静态矩乘反量化 scale（输入与权重 scale 乘积）。
            quant_bias：INT32 [N] 或 None，静态矩乘累加域偏置。
            weight_scale：BF16 [N]，动态路径使用的逐输出通道权重 scale。
    输出：NPU BF16 Tensor [M,N]，本 rank 的投影部分和，尚未 all-reduce。
        动态路径内部生成 INT8 [M,K] 激活和逐 token scale。
    """
    import torch

    if params.quant == "static":
        q = torch.ops.npu.npu_quantize(
            x, params.input_scale_reciprocal, params.input_offset,
            torch.qint8, -1, False,
        )
        y = torch.ops.npu.npu_quant_matmul(
            q, params.weight, params.deq_scale,
            bias=params.quant_bias, output_dtype=x.dtype,
        )
    else:
        q, scale = torch.ops.npu.npu_dynamic_quant(x)
        y = torch.ops.npu.npu_quant_matmul(
            q, params.weight, params.weight_scale,
            pertoken_scale=scale.flatten(), output_dtype=x.dtype,
        )
    return y


def o_proj_allreduce(x, params, process_group):
    """执行本地 o_proj，然后进行 HCCL AllReduce SUM。

    输入：
        x：NPU BF16 [M,K]；params：量化参数包，字段见 o_proj()。
        process_group：原生 HCCL ProcessGroup，包含全部 TP rank。
    输出：NPU BF16 [M,N]，每个 rank 均持有完整的归约结果。
    副作用：执行组内通信；所有 rank 须按相同顺序调用。
    """
    import torch.distributed as dist

    y = o_proj(x, params)
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=process_group)
    return y


def partition_rank_cpus(allowed_cpus, local_world_size):
    """按物理核分组后划分互不重叠的 rank CPU 集合。

    输入：allowed_cpus 为当前 cpuset 允许的逻辑 CPU ID 序列；
        local_world_size(int) 为本机 worker 数量。
    输出：list[list[int]]，下标为 LOCAL_RANK，同一物理核的可用 SMT 线程
        分配给同一个 rank。sysfs 拓扑不可读时退化为逻辑 CPU 划分。
    异常：可分配核组不足时抛出 RuntimeError，不静默让多个 rank 共用 CPU。
        这里只隔离 CPU，不推断 NPU/NUMA 亲和关系、不绑定内存。
    """
    cores = {}
    for cpu in sorted(allowed_cpus):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError):
            key = ("logical", cpu)
        cores.setdefault(key, []).append(cpu)
    groups = list(cores.values())
    if local_world_size < 1 or len(groups) < local_world_size:
        raise RuntimeError(
            f"CPU affinity needs >= {local_world_size} available core groups, got {len(groups)}; "
            "expand the container cpuset or use --no-cpu-affinity with external binding"
        )
    return [
        [cpu for group in groups[len(groups) * rank // local_world_size:
                                 len(groups) * (rank + 1) // local_world_size] for cpu in group]
        for rank in range(local_world_size)
    ]


def bind_rank_cpus(disabled):
    """在导入 Torch、初始化 NPU/通信线程前绑定当前 worker 的 CPU。

    输入：disabled(bool)，True 时保留外部绑核；从 torchrun 环境读取
        LOCAL_RANK、LOCAL_WORLD_SIZE，从 sched_getaffinity 读取实际允许 CPU。
    输出：当前进程绑定后的 CPU ID 列表，供结果 JSON 记录。
    副作用：默认限制当前进程 CPU，后续创建线程继承该限制；打印每 rank 映射。
        设置未指定的 OMP_NUM_THREADS/MKL_NUM_THREADS 为 1，减少线程争用。
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    allowed = sorted(os.sched_getaffinity(0))
    if not disabled:
        bindings = partition_rank_cpus(allowed, local_world)
        os.sched_setaffinity(0, bindings[local_rank])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    assigned = sorted(os.sched_getaffinity(0))
    print(f"[rank={os.environ['RANK']} local_rank={local_rank}] cpu_affinity={assigned}", flush=True)
    return assigned


def init_distributed(tp_size):
    """绑定本地 NPU，并建立原生 HCCL WORLD 通信组。

    输入：tp_size(int)，预期 TP 进程数，必须等于环境变量 WORLD_SIZE。
        从 torchrun 环境读取 RANK、LOCAL_RANK、WORLD_SIZE 及 env:// 所需配置。
    输出：SimpleNamespace，含 rank(int，全局 rank)、local_rank(int，本地设备号)、
        world(int，进程数)、device(torch.device)、group(HCCL ProcessGroup)。
    副作用：设置当前 NPU 并初始化分布式环境；调用方负责销毁通信组。
    异常：WORLD_SIZE 不匹配时抛出 ValueError。
    """
    import torch
    import torch.distributed as dist
    import torch_npu  # Registers the NPU device and HCCL backend.

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != tp_size:
        raise ValueError(f"WORLD_SIZE={world} must equal --tp-size={tp_size}")
    torch.npu.set_device(local_rank)
    # Permit FRACTAL_NZ allocation before constructing any NPU weights.
    torch_npu.npu.config.allow_internal_format = True
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", init_method="env://",
                            world_size=world, rank=rank)
    return SimpleNamespace(rank=rank, local_rank=local_rank, world=world,
                           device=device, group=dist.group.WORLD)


def check_weight_format(weight, requested_format):
    """验证权重的实际 NPU 存储格式，禁止静默退回 ND。

    输入：weight 为 NPU INT8 [K,N]；requested_format(str) 为 nz 或 nd。
    输出：实际格式编号(int)，NZ=29、ND=2；不匹配时抛出 RuntimeError。
    注意：仅在准备/校验阶段调用，不在计时循环内增加格式查询开销。
    """
    import torch_npu

    actual = int(torch_npu.get_npu_format(weight))
    expected = 29 if requested_format == "nz" else 2
    if actual != expected:
        raise RuntimeError(
            f"npu_format_cast failed: requested {requested_format} ({expected}), "
            f"actual format={actual}; refusing to benchmark a different layout"
        )
    return actual


def create_inputs_and_params(args, rank, device):
    """构造可复现、各 rank 不同的合成输入及量化参数。

    输入：
        args：parse_args() 返回的 Namespace，使用尺寸、quant、weight_format 字段。
        rank(int)：全局 rank，用于随机 seed=1234+rank 和偏置选择。
        device(torch.device)：本 rank 的 NPU 设备。
    输出：(x, params, x_cpu, w_cpu) 四元组：
        x：NPU BF16 [M,K]；x_cpu：相同数值的 CPU BF16 [M,K]。
        w_cpu：CPU INT8 [N,K]，取值为 [-3,3] 的整数，用于参考矩乘。
        params：o_proj() 所需 SimpleNamespace；weight 是 w_cpu 转置后的
            NPU INT8 [K,N]，可转换 NZ；其余字段类型和形状见 o_proj()。
            合成输入 scale=0.25，权重 scale=0.125，deq_scale=0.03125，
            offset=0；quant_bias 默认 None；--quant-bias 才在 rank0 传全零偏置。
    注意：这些 scale 是合成数据约定，不来自真实 checkpoint。
    """
    import torch
    import torch_npu

    k = args.num_heads * args.v_head_dim // args.tp_size
    generator = torch.Generator().manual_seed(1234 + rank)
    w_cpu = torch.randint(-3, 4, (args.hidden_size, k),
                          dtype=torch.int8, generator=generator)
    weight = w_cpu.T.contiguous().to(device)
    weight = torch_npu.npu_format_cast(weight, 29 if args.weight_format == "nz" else 2)
    actual_format = check_weight_format(weight, args.weight_format)
    if rank == 0:
        print(f"weight logical_shape={list(weight.shape)}, actual_npu_format={actual_format}", flush=True)
    # Plain tensor bundle, no framework layer or weight loader.
    params = SimpleNamespace(
        quant=args.quant, weight=weight,
        weight_scale=torch.full((args.hidden_size,), 0.125,
                                dtype=torch.bfloat16, device=device),
        input_scale_reciprocal=torch.full((k,), 4.0,
                                         dtype=torch.bfloat16, device=device),
        input_offset=torch.zeros(k, dtype=torch.bfloat16, device=device),
        deq_scale=torch.full((args.hidden_size,), 0.03125,
                             dtype=torch.float32, device=device),
        quant_bias=(torch.zeros(args.hidden_size, dtype=torch.int32, device=device)
                    if args.quant_bias and rank == 0 else None),
    )
    x_cpu = torch.randint(-3, 4, (args.tokens, k),
                          generator=generator).to(torch.bfloat16) * 0.25
    x = x_cpu.to(device)

    return x, params, x_cpu, w_cpu


def compute_reference(x, x_cpu, w_cpu, params, group):
    """生成用于校验的投影参考结果，不计入性能测量。

    输入：
        x：NPU BF16 [M,K]，提供目标设备，动态路径还用它计算激活量化结果。
        x_cpu：对应 CPU BF16 [M,K]；w_cpu：CPU INT8 [N,K]。
        params：量化参数包，此处读取 quant；scale 数值按数据构造函数的约定。
        group：包含全部 TP rank 的 HCCL ProcessGroup。
    输出：NPU BF16 [M,N]，每个 rank 均持有完整归约结果；
        CPU FP32 矩乘后先模拟本地 BF16 舍入，
        再通过 FP32 HCCL SUM 归约并转换为 BF16 的参考结果。
    注意：动态路径复用 NPU 动态量化；矩乘在 CPU 上独立计算。
        FP32 参考归约与基线 BF16 归约可能存在舍入差异。
    副作用：执行一次跨 rank collective，组内须一致调用。
    """
    import torch
    import torch.distributed as dist

    device = x.device
    # Independent CPU GEMM oracle; reduction includes local BF16 rounding.
    if params.quant == "static":
        q_cpu = (x_cpu.float() / 0.25).round().clamp(-128, 127)
        local_ref = (q_cpu @ w_cpu.float().T) * 0.03125
    else:
        q, scales = torch.ops.npu.npu_dynamic_quant(x)
        local_ref = (q.cpu().float() @ w_cpu.float().T)
        local_ref *= scales.cpu().float().reshape(-1, 1) * 0.125
    reference = local_ref.to(torch.bfloat16).float().to(device)
    dist.all_reduce(reference, group=group)
    reference = reference.to(torch.bfloat16)
    return reference


def check_output(actual, expected):
    """校验输出的数值、形状、dtype 和设备是否符合预期。

    输入：actual、expected 均为 BF16 [M,N] Tensor，通常位于同一 NPU。
    输出：None；使用 rtol=0.02、atol=0.5 比较，失败抛出 AssertionError。
    """
    import torch

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5)


def load_implementation(spec):
    """选择内置基线或加载外部融合实现。

    输入：spec(str|None)，None 使用基线，否则格式为 'module:function'。
    输出：可调用对象 fn(x, params, process_group)，参数格式同
        o_proj_allreduce()，返回完整归约后的 NPU BF16 [M,N]。
    副作用：外部 module 会被导入；所有 rank 应使用同一个入口。
    """
    if spec is None:
        return o_proj_allreduce
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def warmup(step, iterations):
    """重复执行以完成编译、初始化及预热，最后等待 NPU 完成。

    输入：step 为无参 callable，通常返回 NPU BF16 [M,N]；graph.replay
        也可传入，其返回 None。iterations(int) 为正整数预热次数。
    输出：最后一次 step() 的返回值（Tensor 或 None）。
    副作用：执行实际计算/通信，并同步当前设备。
    """
    import torch

    for _ in range(iterations):
        output = step()
    torch.npu.synchronize()
    return output


def build_module(args, params, process_group):
    """构造 eager 投影通信模块，保留已转换的权重存储。

    输入：args 为命令行 Namespace，读取 impl 和 weight_format；
        params 为量化 tensor 参数包；process_group 为 TP HCCL ProcessGroup。
    输出：eval 模式的 nn.Module，
        调用签名为 module(x: BF16 [M,K]) -> BF16 [M,N]。
    副作用：检查 Module 权重仍是指定格式；不使用 torch.compile 或 TorchAir。
    """
    from glm52_o_proj_module import OProjAllReduce

    module = OProjAllReduce(params, process_group, load_implementation(args.impl)).eval()
    check_weight_format(module.weight, args.weight_format)
    return module


def capture_graph(step, warmup_iterations):
    """在独立 NPU stream 上捕获计算图并预热 replay。

    输入：step 为无参 callable，返回 NPU BF16 [M,N]，需支持图捕获；
        warmup_iterations(int) 为正整数，表示捕获后的 replay 预热次数。
    输出：(graph, output)：graph 为 torch.npu.NPUGraph；output 为捕获时
        返回的 BF16 [M,N] Tensor，其存储由后续 replay 更新，须保持引用。
    副作用：执行 capture 和通信、stream 等待及设备同步；所有 rank 一致调用。
    """
    import torch

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = step()
    torch.npu.current_stream().wait_stream(stream)
    warmup(graph.replay, warmup_iterations)
    return graph, output


def create_profiler(profile_dir, rank):
    """创建测量循环使用的 Ascend Profiler 上下文。

    输入：profile_dir(str|None)，采集根目录，为空时关闭采集；rank(int)，
        用于将各进程的输出分到 rank_<rank> 目录。
    输出：上下文管理器；启用时进入上下文返回 profiler 对象，采集 CPU/NPU
        和 Level1 通信数据；关闭时为 nullcontext，进入后返回 None。
    副作用：启用后在退出上下文时解析并写入性能数据。
    """
    import torch_npu

    profiler_context = nullcontext()
    if profile_dir:
        profiler_context = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(Path(profile_dir) / f"rank_{rank}")),
        )
    return profiler_context


def benchmark(step, iterations, group, profiler_context):
    """对重复计算进行采集和端到端计时。

    输入：
        step：无参 callable，eager 返回 BF16 [M,N]，graph.replay 返回 None。
        iterations(int)：正整数测量次数。
        group：HCCL ProcessGroup，用于测量前 barrier 对齐各 rank。
        profiler_context：create_profiler() 返回的上下文管理器。
    输出：(output, elapsed)：output 为最后一次 step() 的返回值；
        elapsed(float，秒) 为全部迭代的总时间，包含 Python 提交和末尾设备同步，
        不含采集启动后的准备轮次、barrier 和 profiler 建立/退出。
        Graph 输出另由调用方保留。准备阶段以 benchmark_setup 标记，分析时排除。
    副作用：执行计算/通信、同步；启用 profiler 时每轮调用 prof.step()。
    """
    import torch
    import torch.distributed as dist

    dist.barrier(group=group)
    torch.npu.synchronize()
    with profiler_context as prof:
        # Profiler startup differs across workers. Warm its first-call paths, then align.
        with torch.autograd.profiler.record_function("benchmark_setup"):
            if prof is not None:
                warmup(step, 3)
            torch.npu.synchronize()
            dist.barrier(group=group)
            torch.npu.synchronize()
        # No per-iteration barriers: retain real steady-state rank skew in the trace.
        start = time.perf_counter()
        for _ in range(iterations):
            output = step()
            if prof is not None:
                prof.step()
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
    return output, elapsed


def save_results(args, runtime, x, w_cpu, params, candidate, elapsed):
    """汇总各 rank 耗时并保存尺寸、运行模式和校验状态。

    输入：
        args：命令行 Namespace，提供迭代次数、运行模式及 output_dir。
        runtime：init_distributed() 返回的设备、rank、通信组信息。
        x：NPU BF16 [M,K]；w_cpu：CPU INT8 [N,K]；params：量化参数包，
            通过 params.weight 记录 NPU 权重逻辑形状 [K,N]。
        candidate：已通过校验的 NPU BF16 输出 [M,N]。
        elapsed(float，秒)：本 rank 全部测量迭代的总耗时。
    输出：None；写入 output_dir/rank_<rank>.json，rank0 同时打印结果。
        us_per_iteration 为本 rank 平均微秒数；max_rank_us_per_iteration
        为各 rank 平均耗时的最大值。correctness='passed' 依赖调用前校验。
    副作用：测量区外执行 HCCL MAX 归约，并创建目录、写文件。
    """
    import torch
    import torch.distributed as dist

    # Aggregate after profiling, never inside the measured loop.
    actual_format = check_weight_format(params.weight, args.weight_format)
    latency = torch.tensor(elapsed / args.iterations * 1e6, device=runtime.device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=runtime.group)
    result = dict(rank=runtime.rank, tp_size=runtime.world, tokens=args.tokens,
                  input_shape=list(x.shape), weight_logical_shape=list(w_cpu.shape),
                  weight_runtime_shape=list(params.weight.shape),
                  weight_format=args.weight_format,
                  collective="all-reduce", quant_bias=args.quant_bias,
                  output_shape=list(candidate.shape), quant="W8A8" if args.quant == "static" else "W8A8_DYNAMIC",
                  graph=args.graph, impl=args.impl or "o_proj_allreduce",
                  compiled=False, weight_actual_npu_format=actual_format,
                  cpu_affinity=runtime.cpu_affinity,
                  cpu_affinity_enabled=not args.no_cpu_affinity,
                  iterations=args.iterations, us_per_iteration=elapsed / args.iterations * 1e6,
                  max_rank_us_per_iteration=latency.item(),
                  profiler_enabled=bool(args.profile_dir), correctness="passed",
                  device=torch.npu.get_device_name(runtime.local_rank))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank_{runtime.rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    if runtime.rank == 0:
        print(json.dumps(result, indent=2), flush=True)


def run(args):
    """串联初始化、数据构造、eager Module、精度校验、预热、采集及保存。

    输入：args 为 parse_args() 返回的 Namespace；执行环境由 torchrun 提供。
        输入/输出 tensor 格式遵循模块开头的 M、K、N 约定。
    输出：None；每 rank 写入结果 JSON，可选写入 profiler 数据。
    副作用：导入 Torch 前按 rank 绑核，再在 inference_mode 下运行计算和通信；正常返回或异常
        退出时均销毁已建立的通信组。Graph 模式保留并校验 replay 输出。
        不使用 torch.compile；格式转换和检查不计入采集区间。
    """
    cpu_affinity = bind_rank_cpus(args.no_cpu_affinity)

    import torch
    import torch.distributed as dist

    runtime = init_distributed(args.tp_size)
    runtime.cpu_affinity = cpu_affinity
    try:
        with torch.inference_mode():
            x, params, x_cpu, w_cpu = create_inputs_and_params(
                args, runtime.rank, runtime.device)
            reference = compute_reference(x, x_cpu, w_cpu, params, runtime.group)
            baseline = o_proj_allreduce(x, params, runtime.group)
            check_output(baseline, reference)

            module = build_module(args, params, runtime.group)
            step = partial(module, x)
            check_output(step(), baseline)
            warmup(step, args.warmup)
            check_weight_format(module.weight, args.weight_format)

            graph = None
            if args.graph:
                graph, graph_output = capture_graph(step, args.warmup)
                check_output(graph_output, baseline)

            profiler = create_profiler(args.profile_dir, runtime.rank)
            candidate, elapsed = benchmark(
                graph.replay if graph is not None else step,
                args.iterations, runtime.group, profiler)
            if graph is not None:
                candidate = graph_output
            check_output(candidate, baseline)
            save_results(args, runtime, x, w_cpu, params, candidate, elapsed)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    if args.describe:
        print(json.dumps(dict(input=[args.tokens, args.num_heads * args.v_head_dim // args.tp_size],
                              weight=[args.hidden_size, args.num_heads * args.v_head_dim // args.tp_size],
                              matmul_output=[args.tokens, args.hidden_size],
                              output=[args.tokens, args.hidden_size], collective="all-reduce",
                              tp_size=args.tp_size), indent=2))
    else:
        run(args)
