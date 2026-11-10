# V4.1 HCCL Graph 通信顺序配置

本文说明已实现的软件配置接口。完整 V4.1 模型的资格门禁仍关闭；以下配置不能
绕过门禁，也不代表原生模型、DSpark、PD 或性能已验收。

在现有 `additional_config` 的 `afd.connector_extra_config` 中指定顺序，保留
其余角色、拓扑与连接参数。例如该配置片段选择跨层顺序：

```json
{
  "afd": {
    "connector_extra_config": {
      "graph_exchange_order": "cross_layer_v1"
    }
  }
}
```

| 值 | 含义 |
| --- | --- |
| `stage_serial` | 默认顺序；各 stage 顺序交换。省略字段等价于此值。 |
| `bulk_v1` | U2 同层按约定的输入前沿交换。 |
| `cross_layer_v1` | U2 使用固定的跨层交换顺序，并在末层排空输出。 |

`max_tokens` 仍可与此字段同时配置。字段值区分大小写；其他值、`null`、数字和
未知字段都会拒绝。默认配置的序列化保持原来的 `max_tokens` 映射，不新增顺序
字段；默认 HELLO 也保持原有编码。

非默认顺序只用于严格的 `FULL_DECODE_ONLY` Graph 配置；eager 或 legacy 入口
会在建立通信域前拒绝。当前非默认顺序要求 A/F 等配对、原生 TP1 或 TP2。
A 和 F 必须配置同一个值，F 的角色隔离会保留该字段。
两端在 HELLO 中核对顺序，本地模型绑定也必须与已协商的顺序一致。

选择顺序不会自动打开 compute streams。两种非默认顺序都需要实际 U2 compute
pipeline；`cross_layer_v1` 还要求 A 的 hybrid DAG、F 的 receive stream 和
cross-layer 条件成立。现有 overlap 开关和所有 peer 的捕获预检仍决定相应 U2
pipeline 是否可捕获；条件不足时共同跳过捕获。Attention 三流配置不代表图内
HCCL 可以移到任意通信侧流。U1 和已协商
session 内的 eager fallback 使用顺序 tensor 路径，不能据此认为非默认顺序失效。

顺序在 session/cache 的存续期间固定。修改顺序后，所有 A/F ranks 应使用新进程、通信
域和 Graph cache；不要在已有 session 内切换顺序或复用另一顺序的捕获结果。

实际 A5 组件验证继续使用
[批量验证说明](DSV41_A5_BATCH_VALIDATION_ZH.md)；其中各测试独立创建 session。
这些受控组件与正常模型启动、真实接受/拒绝、KV/history 和性能分别验收。
