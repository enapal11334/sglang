"""以 eager 模式执行、保留 NZ 权重的 W8A8 o_proj + AllReduce 模块。"""

import torch


class OProjAllReduce(torch.nn.Module):
    """持有投影参数和 HCCL 通信组，forward 返回完整 TP 归约结果。"""

    def __init__(self, params, process_group, implementation):
        """注册量化参数为 buffer，避免训练参数和外部参数包参与调用。

        输入：params 为数据构造函数返回的 SimpleNamespace，含 quant(str)、
            weight(INT8 [K,N])、weight_scale(BF16 [N])、
            input_scale_reciprocal/input_offset(BF16 [K])、deq_scale(FP32 [N])、
            quant_bias(INT32 [N] 或 None)，所有 tensor 已位于当前 NPU。
            process_group：TP HCCL ProcessGroup。
            implementation：fn(x, params, process_group)，本模块将自身作为参数包传入。
        输出：无；初始化 nn.Module，buffer 保留原有 NZ/ND 物理存储。
        """
        super().__init__()
        self.quant = params.quant
        self.process_group = process_group
        self.implementation = implementation
        for name in (
            "weight", "weight_scale", "input_scale_reciprocal",
            "input_offset", "deq_scale", "quant_bias",
        ):
            self.register_buffer(name, getattr(params, name))

    def forward(self, x):
        """执行 W8A8 本地投影与跨 rank AllReduce SUM。

        输入：x 为当前 NPU 上的 BF16 [M,K]，K 是本 rank 的 attention 输入宽度。
        输出：当前 NPU 上的 BF16 [M,N]，各 rank 均持有完整归约结果。
        副作用：组内所有 rank 必须按相同顺序调用通信。
        """
        return self.implementation(x, self, self.process_group)
