# 各 A 实际计数汇总：受控实现契约（2026-11-09）

本记录是实现者的契约与本地检查说明，尚待独立审查及实际依赖/NPU 探针；不是硬件验收。生产 guard 是否打开由后续 native A/F 生命周期集成决定。

## 冻结接口及 wire

新增 CPU-safe 类型位于 `afd_plugin/distributed/hccl_plan.py`：

```python
@dataclass(frozen=True, slots=True)
class HcclLocalForward:
    mode: str
    valid_tokens: tuple[int, ...]
    padded_tokens: tuple[int, ...]

def submit_forward(
    self,
    local: HcclLocalForward,
    *,
    transaction_id: str | None = None,
) -> HcclParentPlan | None: ...

def stop_attention(self) -> None: ...
```

两个方法属于 `HcclStagedTransport`。所有 A 调用 `submit_forward`；A0 提供 transaction ID，其他 A 可以省略，若提供则必须匹配请求。所有 A 在正常停机时调用 `stop_attention`。F 继续 `receive_forward`，不提交 A 的计数。

`HcclPlanCapabilities.local_counts_version` 默认 0，baseline HELLO 不编码新字段，保留既有 v3 字节。显式启用为 1 时所有 A/F 必须匹配，协商不降级；旧 strict-v3 读到扩展字段会在 HELLO 完整帧阶段拒绝，新 peer 读到旧能力 0 也拒绝。v2 未改。能力 1 仅允许 `attention_size == ffn_size == native_tp_size`，即本轮 native DP1 的每个角色恰好一个 TP world；未支持的组合在 HELLO 前校验/协商阶段失败。能力 0 原有 TP1 不等拓扑保留。

新控制消息继续使用有界长度前缀和 strict v3 JSON envelope：

- `COUNTS_REQUEST`：仅 `transaction_id`，由 A0 发给其他 A。
- `COUNTS_REPORT`：`transaction_id`、实际发送者 `world_rank`、`local_forward={mode, valid_tokens, padded_tokens}`。

局部报告恰好一或两 stage，数组等长、严格整数且不接受 bool，`0 <= valid <= padded <= INT32_MAX`。transport 再验证协商的 stage/token capacity。单条帧仍不超过 65536 bytes，错误文本仍不超过 4096 UTF-8 bytes。报告只携带 native counts，不携带伪造 peer offsets；父布局由现有 `HcclStageLayout` 推导。

## 顺序、核对及失败

```mermaid
sequenceDiagram
    participant A0
    participant A1 as Other A ranks
    participant F as All F ranks
    A1->>A1: wait for REQUEST or STOP
    A0->>A1: COUNTS_REQUEST(id)
    A1->>A0: COUNTS_REPORT(actual local stages)
    A0->>A0: drain all reports; verify mode/U/TP counts
    A0->>F: FORWARD(full plan)
    A0->>A1: FORWARD(full plan)
    A1->>A1: compare own plan column with saved report
    F->>A0: ACCEPT(id,digest,rank)
    A1->>A0: ACCEPT(id,digest,rank)
    A0->>F: global ACCEPT
    A0->>A1: global ACCEPT
```

A0 把自己的报告与实际收到的每个 A 报告按 role rank 放入完整计划。每组 native TP 的 valid 和 padded 都必须逐 stage 相同；不能从 A0 复制出全组 counts，也不能只核对总数。mode、stage 数不一致同样拒绝，不在计数汇总后静默把 U2 改成 U1。每个 A 在发送 ACCEPT 前复核 parent ID/mode、stage 数及自己的 valid/padded 列；FORWARD 摘要 ACK 仍绑定完整父计划。

所有已请求 A 的完整报告帧必须先排空，再向全 world 发送 ERROR。完整但 JSON/字段无效、身份不符、模式/TP counts 不符、局部报告失败均属于此语义拒绝。F 此时仍等待 FORWARD/STOP，可以接收同一个全局 ERROR。peer 收到完整无效 coordinator REQUEST/FORWARD 也报告 ERROR，并等待全局拒绝，避免 A0 一直等 ACCEPT。

帧长度损坏或 backend I/O RuntimeError 不能保证 sender 已结束阻塞发送，直接标记 terminal，交 backend timeout/进程监督；不尝试向仍在 send 的 peer 发送伪造 ERROR/abort/STOP。模型/设备异常发生在汇总前或 payload 中时也仍需外部监督，计数 API 没有把它们变成可恢复的网络取消。

计数轮开始后，`_counts_transaction_id` 阻止 idle STOP；成功后的父计划继续阻止 STOP，直到所有输出与全体 COMPLETE 完成。`_clear_plan` 清除局部报告和计数轮 ID，允许下一次父事务。重复 ID/会话上限由现有 replay 保护拒绝；失败会话需要重建。

其他 A 的入口先接收请求，收到 idle STOP 时只 STOP_ACK、等待 global STOP_ACK 并返回 None。正常 `stop_attention` 对能力 0/1 都使用相同握手；A0 广播，其他 A 仅等 STOP，误入其他阶段则拒绝并 terminal。不能在 A1 入口先主动 blocking send 报告，否则 A0 选择 STOP 时会互等。

## U1/U2 与状态归属

同一个能力上限 2 的 v3 session 可以依次接受一 stage 与两 stage 父计划，不重新创建 stage group；COMPLETE 的 device synchronization 和清理是跨父边界。U1 的意思是一 stage v3 父事务，不能在同一 transport 中落回 legacy v2 wire。profile/dummy/target 模式仍由 native execution scope 提供，本模块不重建输入、历史或设备状态。

实际本地计数必须由 native `InputBatch` 和实际 `UBatchState.slices` 推导，不由配置 `num_ubatches=2` 猜测。锁定 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607:vllm/v1/worker/gpu/ubatch_utils.py:_slice_input_batch` 使用 `padded = stop-start`、`valid = max(0,min(stop,input_batch.num_tokens)-start)`；dummy 的 padding mask 与 native token count 有其既定意义，不能通过额外设备读 mask 偷换计数定义。本轮 collector 只收集/验证 caller 提供的实际报告，native adapter 的推导与调用由独立任务审查。

没有更改 V4.1 HC、pre_mix、Engram 生产者/退役、权重、FP32 router 精度或 native TP reduction，也不创建/销毁 native TP group。新增成本是每父事务额外 `2*(A-1)` 条 Gloo 控制消息（各含长度和正文），A0 `O(A*U)` 的 Python 整数布局；TP1 时没有额外远端 counts 消息。没有新增 NPU tensor copy 或 HCCL payload，但增加同步控制往返，不宣称 overlap 或性能提升。

## 检查及快照

本地：`pytest tests/unit/distributed/test_hccl_plan.py tests/unit/distributed/test_hccl_session.py -q` 退出 0，154 项通过；Ruff check/format、Python 3.11 AST 和 `git diff --check` 通过。transport 镜像测试因本机缺 torch/vLLM/torch_npu 跳过，已交 A3 实际依赖回归；不能将跳过写成通过。

测试保护 codec roundtrip/strict capability、actual TP reports、每 A 的 ACCEPT 前核对、错误 frame/identity/mode/U/valid/padded/capacity、所有报告先 drain 再 ERROR、I/O terminal、STOP 先于报告、同会话 U1/COMPLETE/U2/COMPLETE/STOP。共享 `ScriptedStagedControlPeer` 只替代外部 control I/O，进出经过真实 codec，断言 wire transcript；没有模拟 NPU/HCCL tensor 行为。CPU 同会话案例使用全空父事务，实际活跃 stage/设备通信仍待探针。

冻结 SHA256（独立审查及 A3 回归前）：

| 文件 | SHA256 |
| --- | --- |
| `afd_plugin/distributed/hccl_plan.py` | `b3da12187d1355b05630f9167b1da198bbb9b8a18b25e7fc7c13e675983128b2` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `76878b3103a31338cb88033f4315b99ca33371183cc1ea4942c7b2c6d4d6d344` |
| `tests/unit/distributed/test_hccl_plan.py` | `661e5110e6b983c2a3c42f09100dedab49ebaa0d5d0ccb3cc6858d9902a3754d` |
| `tests/unit/connectors/npu/test_hccl_staged_transport.py` | `789eca84e21499d0f8d6b8e97cd77f6f37d88026c43198bc5818da63ea8d698c` |

移除/演进条件：由原生执行控制面提供每 rank 的实际 stage counts 与统一接受/停止协议后，可移除本地扩展。DP>1/不等拓扑需要先定义独立 TP world 的身份、父输入及 F native context，不能只放宽本轮 capability 校验。
