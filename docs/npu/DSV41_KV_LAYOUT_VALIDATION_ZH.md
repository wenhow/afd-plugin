# V4.1 KV 布局只读诊断

此工具比较原生 KV 配置与实际 tensor views 的布局，并检查一次请求的
source/destination block IDs。它不会启动模型、注册 Mooncake buffer、传输 KV，
也不会放开现有 PD / Graph / DSpark 组合门禁。

## 从已有运行进程采集

采集函数接收原生 worker 已经持有的 `KVCacheConfig` 和分配后的
`kv_caches` 映射。只有持有这两个真实对象的调用方才能执行以下代码；当前没有新增
HTTP/RPC 自动采集入口，不应把示例当成可从外部直接运行的模型启动命令。

```python
from pathlib import Path

from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout

snapshot = collect_deepseek_v41_kv_layout(kv_cache_config, kv_caches)
Path("kv-layout.json").write_text(snapshot.to_json(), encoding="utf-8")
```

`kv_cache_config` 必须是锁定原生类型，`kv_caches` 必须包含全部 descriptor
命名层的实际 materialized tensor。初始范围为 V4.1 `layer_stride=0` 的
shared pages，包括 long KV、index K/scale、compressor ring、SWA，以及实际配置中
存在的 draft SWA 别名。支持 CPU 真实 views 用于接口测试和 NPU 实际 views 的只读
元数据采集；CPU 结果不证明 NPU allocator 或 PD 正确。空池、meta、host-resident、
非 V4.1 spec、未支持的 descriptor 布局会拒绝。

采集只读取 shape、dtype、byte stride、storage 大小及地址差来重建别名关系；
输出不包含绝对地址，不读取 KV 数值，不同步设备，也不保留 tensor 引用。
两端应在本地原生注册配置和 views 均确定后采集。两份本地快照不能证明网络中的
远端 metadata 正是这份快照。

## 比较快照

在拥有 afd-plugin 的 Python 环境运行，比较 CLI 仅依赖标准库：

```bash
python -m afd_plugin.compat.npu.kv_layout prefill-kv-layout.json decode-kv-layout.json
```

退出码 `0` 表示这些快照的全部登记布局字段一致，`1` 表示布局不同，`2` 表示
输入损坏或请求 block 映射非法。JSON 输出列出差异、两端容量以及是否校验请求映射。
绝对地址、allocation 的不透明前缀和本地 `num_blocks` 不作跨进程等值比较；
组件顺序、spec/group flags、dtype、非 block 维 shape、byte strides、相对 offset、
完整物理 page 的 alias 关系必须相同。不同 descriptor 的 page pools 可以在同一
allocation 中互不重叠，跨端不比较这类 slab 距离；不同 descriptor 的部分重叠会在
本地拒绝。容量不同可以合法，
每份快照自身的 tensor/span/page/descriptor 必须仍在自己的 storage 内。

## 校验一次请求的 block 映射

两个 JSON 文件分别是按原生 `transfer_group_ids` 顺序排列的数组，每个元素是该
group 的完整 block ID 数组。不得省掉后面的 group、截断数组或用本地容量代替布局。
例如三组的映射文件可以是 `[[2, 1], [0], []]`，目标可以是 `[[6, 4], [5], []]`。

```bash
python -m afd_plugin.compat.npu.kv_layout \
  prefill-kv-layout.json decode-kv-layout.json \
  --source-blocks prefill-request-blocks.json \
  --destination-blocks decode-request-blocks.json
```

只有两端布局一致、group 数量完整、各组 source/destination 数量相同且每个整数
ID 都在对应本地池内才成功。对同一个 physical page pool，多个 logical groups 或
同一 group 不得把不同 source block 映射到同一个 destination block；一致重复的
pair 允许。尾部一个非法 ID 或冲突映射也会拒绝整个请求。这个检查不改变原生
block 所有权、分配或传输，不证明这些 ID 当前属于相应请求，也不验证 KV 数值、
compressor/history/Engram 交接或实际完成通知。

建议回传两份 `kv-layout.json`、对应请求映射（如有）和 CLI 的完整输出/退出码。
真实 P/D 冷启动、跨进程注册契约、请求完成/失败收尾及模型结果仍需独立验收。
