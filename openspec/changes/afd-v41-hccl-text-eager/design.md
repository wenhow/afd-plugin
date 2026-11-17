# V4.1 文本 eager 的角色模型适配：技术设计

状态：角色模型、V2 计划接入和均匀 TP 配对已实现；A3/A5 组件验证通过，完整 checkpoint 尚未验收。A5 8×96 GiB 的下一步为原生 TP8 控制组，AFD 容量候选为 A4/F4、两侧各 TP4；峰值仍待实测，进度以 tasks.md 为准。
源码标号见 [锁定阅读索引](../../../.agents/dsv41/sources.lock.json)。
锁定签名、实际返回与注入限制见 [原生契约记录](evidence/native-contract.md)。
## 原生调用链与阅读入口

U01 的 AscendDeepseekV41ForCausalLM（VL wrapper）
→ U02 的 AscendDeepseekV41LLMForCausalLM
→ DeepseekV41Model → DeepseekV41DecoderLayer → DeepseekV41MoE。
wrapper 即便文本也声明 requires_raw_input_tokens=True，get_model_state_cls 转发至语言模型。
优先复用 language_model_cls、model_cls、decoder_layer_cls，实际签名以锁定代码为准。

## A/F 切分

A 保留 token、Engram、attention/shared index/KV、HC/归一化和最终 collapse。
在 decoder 构造时把 mlp 设置为无参数远程 proxy，保持原生 forward。
原生 decoder 没有 mlp_cls hook，直接构造完整 MoE；A 不能先 super 构造后再删除专家。
需审查最小 decoder constructor override，并明确原生 set_moe_parameters 忽略 proxy 后，
A 无本地专家的元数据如何供 runner/loader 使用；不得伪造本地专家来通过扫描。
residual、ffn_post、ffn_comb 属于本层 A 的栈/continuation，ffn_pre 返回成为下一层 pre_mix；不额外把整组 HC tensor 发往 F。

F 只创建实际 MoE 层，含 router、routed/shared experts，按完整命名空间加载 .ffn./.mlp. 权重。
F 不需要 VL wrapper 的视觉模块、Engram 大表、attention/KV、输出 head；避免完整构造再删除。
保持原生共享专家返回/合并、routed_scaling_factor、SP/TP 归约语义。

proxy 必须接收 hidden_states、input_ids、hidden_states_fp32、already_sequence_parallel，并正确映射布局。
first path 的 gate 在 F，不在 A 端提前算 router；hash 层使用准确 IDs。

## 精度与状态

逐设备核对 post_attention_layernorm 和 DeviceOperator.rmsnorm 的两路返回。
若 FP32 是额外精度的结果，必须单独传输；若能证明等于已发送 tensor.float()，才选择 F 重建。
记录元素级误差、router top-k 和最终输出差异。不能把单设备结果推广到所有后端。

实际 A3 fallback 的 18 组 FP32 与 BF16.float 相等；融合 norm/custom top-k 缺编译扩展，
仍保留 native FP32 返回的独立传输，不据此统一改为重建。
量化模型必须显式 `--quantization deepseek_v4_fp8`；锁定版本的自动选择实际得到 generic FP8，
loader 拒绝该错误格式。当前 A3/CANN 9.0.1 的实际 DynamicMxQuant 调用报不支持，
不能靠增加卡数解除。随后隔离安装正式 CANN 9.2.0，实测仍拒绝该 A3 路径。
见 [精度证据](evidence/precision-validation.md)与 [9.2.0 复测](evidence/cann-920-validation.md)。

Engram 的 request history/lookback、入队、释放和 dummy 输入沿用原生 V2 ModelState；
shared attention state 的 source/consumer 生命周期沿用原生 model forward。
任何局部上下文在连续请求之间必须清理，不能用新可变全局表保存。

## 回退与验证

V4.1 专用路径独立注册，撤销该路径可回到原 v0.30 模型集。
测试覆盖角色权重归属、原始 IDs、proxy 输入语义和本地 HC 状态；
真正 HC/Engram/量化算子和精度验收在目标 NPU完成，不用 fake 模拟其正确性。
先原生 text eager 控制组，再相同 checkpoint、tokenizer、prompt、采样参数的 AFD。
资源不足先调整 A/F 规模并说明，不能默默更换 checkpoint 或关闭 Engram。

## 均匀 TP 配对

要求 A/F rank 数均等于各角色 native TP size，每个 A_i 与 F_i 使用 base_port+i 独立 pair。
HELLO v2 核对 native_tp_size/pair_rank；PP/DP/PCP/DCP=1，EP/SP 关闭。
Connector 只传完整 token 行，不增加 TP 归约；F 继续执行 native 完整 MoE。
EngineCore 校验每个 F rank 的严格 bool 状态，全部 STOP 后才正常退出。
pair 关闭不销毁 native TP 组；故障后须重建全部角色进程。
[真实 TP2 组件](evidence/tp-validation.md)已验证该组边界，尚未验证 native MoE 或 TP8。
资源候选还需计入 [实际 RoPE 常驻与 Engram 重编码](evidence/resource-budget.md)，不能只按磁盘权重均分。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
“保留原生 forward、替换 MoE”是本 change 同步 U1 的边界；C4/C6 为异步执行增加目标 V4.1 continuation，并单独审查，不将旧 V4 tuple/aux 时机照搬。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
