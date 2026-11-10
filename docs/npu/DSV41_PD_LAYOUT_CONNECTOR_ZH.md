# V4.1 PD 布局校验连接器

此连接器通过 vLLM 的外部 KV connector 工厂显式选择，在原生 Mooncake
注册及传输入口增加布局一致性校验。V4.1 的完整模型生产门禁仍关闭；以下是组件配置，
不代表已经完成 NPU PD、请求历史或 Engram 验收。

配置中保留原生连接器名称，并指定插件模块：

```json
{
  "kv_connector": "MooncakeHybridConnector",
  "kv_connector_module_path": "afd_plugin.compat.npu.kv_layout_connector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "prefill": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
    "decode": {"tp_size": 1, "dp_size": 1, "pp_size": 1}
  }
}
```

P 端使用 `kv_producer`，D 端使用 `kv_consumer`，两端均须选择上述模块。
两端原生 TP 大小须相同，DP/PP/PCP/DCP 均为 1。仅支持 eager、无推测解码、
无 prefix cache、无 sparse KV offload、无 sleep 和权重更新的 V4.1 共享页路径。
地址、端口、engine ID 等其他字段沿用部署中的原生 Mooncake 配置。

注册前，连接器从实际 `KVCacheConfig` 和 KV 张量视图采集布局，核对 native
page 顺序、dtype、字节步长、页内偏移、物理页别名及逻辑分组。不可传输、draft、
host-resident 或不受支持的布局会在内存注册及线程启动前拒绝。每个 native engine
仅允许一个注册生命周期；注册或启动部分失败后须重建进程，不能复用已有 engine 重试。

握手必须携带布局字段，旧连接器或布局不兼容的对端会被拒绝。接收端将布局与地址、
页步长、容量及传输端点绑定，按原生窗口尾部投影核对整批 block 映射，之后才调用
native batched read。两端合法容量可以不同；block ID 必须分别落在各端容量内。
这项校验依赖合作端提供的信息，不提供远端身份认证或远端真实注册证明。

排查布局差异时可先使用[只读布局工具](DSV41_KV_LAYOUT_VALIDATION_ZH.md)。
需要分别保留配置、首个错误、各端布局快照和原生 Mooncake 日志，避免将失败后的
新地址或 metadata 与旧注册混用。CPU 接口测试覆盖 codec、注册前检查、地址发布、
窗口尾部映射及失败请求的原生 DONE 收尾；实际设备传输仍须单独验证。
