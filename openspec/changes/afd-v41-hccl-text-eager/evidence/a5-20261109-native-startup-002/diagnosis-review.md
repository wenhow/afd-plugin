# A5 第二次原生启动：独立诊断与量化补丁审查

审查日期：2026-11-09。结论：此次启动停在权重加载后的 DS Linear scale 打包；
最小 odd-pair metadata padding 方案在静态审查中无阻塞发现。
目标 A5 wheel 的真实算子、完整模型启动及请求仍待验证。

## 输入与实际失败阶段

仅通过 Python `tarfile.open(..., "r:gz")`、`extractfile()` 将普通成员读入内存，
未使用 `extract()`/`extractall()`，未执行包内文件。

- 用户原包：`C:/Users/zq/Downloads/238c6abfd7a845eb9ed400ea68e7d275.gz`。
- 原包 SHA256：`0f5b84a8a7ae858f22e72d2f41a87dc4be0b834d78509def887520dfea4317b0`。
- 原日志成员：`dsv41-native-tp8.GzC0bW/native-server.log`，101981 字节。
- 同包 `server-exit-code.txt` 为 `1`。
- 同包 config/index hash 匹配既有预期，48 个 shard 均存在；这不是权重 payload 校验。
- 实际 Python 3.11.10、torch 2.10.0、torch_npu 2.10.0.post4、FastAPI 0.123.10。
- 实际 Ascend HEAD：`2db8549c548731611baeb169fa1ef13aa55fc6aa`。
  包版本为 `0.19.1rc2.dev637+g2db8549c5`，editable 路径 `/opt/vllm-ascend`。
  vLLM 包为 `0.30.0+empty`，该包未给出 vLLM Git SHA。

日志第 17、31 行已解析 Engram 三项为 False；第 433–437 行显示加载 48/48，
权重加载耗时 54.52 秒。最早 worker traceback 从第 444 行开始，实际根异常是第 470–473 行：

```text
.../w8a8_mxfp8.py:661, in process_weights_after_loading
layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
RuntimeError: shape '[5120, 4, 2]' is invalid for input of size 46080
```

调用链为 `WorkerProc -> worker.load_model -> MRV2.load_model ->
BaseModelLoader.load_model -> process_weights_after_loading -> AscendLinearMethod ->
AscendW8A8MXFP8DSDynamicLinearMethod`。API 进程后续 Engine core initialization failed
是传播结果。DeepSelect 缺失和 internal format warning 不是此次最早致命异常。
此包未到 profile、KV 分配、HTTP health 或首次请求，不能证明这些阶段通过。

## 锁定源码与层形状定位

只读 `git show` 对照 Ascend 锁 `7e37e29a7803d6897a7113088e2b771f83205bc0`
和实际 `2db8549c548731611baeb169fa1ef13aa55fc6aa`。以下三个相关文件在两提交中逐字节相同：

| 文件 | SHA256 | 相关符号及行号 |
| --- | --- | --- |
| `vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py` | `ddfad441c01057057903520f67b133871b3ad07cac869a7bb49e317176d300df` | DS Linear 类 609；`process_weights_after_loading(self, layer)` 645–682；base odd packing 205–212 |
| `vllm_ascend/models/deepseek_v41/model.py` | `989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f` | `DeepseekV41MLP` 116；shared MLP 构造 204–220；wo_a/wo_b 438–457 |
| `vllm_ascend/ascend_config.py` | `1a6b5f644dc412c35c112c47cfbf9649fec00af7455a1e1e6dc27be5cee2a676` | `FinegrainedTPConfig` 1354；OTP/MTP 限制 1390–1414 |

checkpoint 的 hidden_size=5120、moe_intermediate_size=2304、n_shared_experts=1，
block size 为 32×32。原生 shared MLP 使用 `RowParallelLinear(2304, 5120)`；
SP 关闭、TP8 时本地权重为 `[5120, 288]`，scale 为 `[160, 9]`，展开后
`[5120, 9]`，元素数恰为 46080。九列不能直接 reshape 为 `[5120, 4, 2]`。
因此最强形状定位是 `model.layers.*.mlp.shared_experts.down_proj`；
traceback 未直接打印 prefix，此处明确属于源码与形状推断。

wo_b 的输入为 `o_groups * o_lora_rank = 8 * 1024`，TP8 本地 K=1024，
有 32 个 scale group；wo_a 的本地 N=1024、K=4096。两者均不匹配 5120×9。
不能用 oproj/mlp finegrained TP 配置回避：锁定配置要求普通 TP1、Graph 和 PD Decode，
不允许本次 TP8 eager 控制组。仅切换 Ascend 到锁定提交也不会修复相同函数。
切换 generic fp8 会改变整个量化方法选择，不能据通用日志建议当作此次局部修复。

## metadata pair 边界与最小修复

逻辑 K=288 已按 32 整除，问题是两个 group scale 组成一个 metadata pair。
完整有效九列须保持顺序，最后一个 pair 的空位补零：
`[5120, 9] -> [5120, 10] -> [5120, 5, 2] -> [5, 5120, 2]`。
权重 `[288,5120]` 和 activation `[M,288]` 保持逻辑 K=288。

上游同文件 base MXFP8 Linear 在第 205–212 行已有相同 odd-column packing。
官方 [QuantMatmul API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)
的 MX 全量化表给出 `x2=(K,N)`、`scale=(ceil(K/64),N,2)`；
官方 [DynamicMxQuant API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)
给出 scale 的 group 轴按两个 group 向上取整、尾轴为 2。
[CANN TCubeTiling 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0673.html)
也描述 GM 输入的 scale group 数为奇数时补零至偶数。
这些官方资料当前版本支持该布局判断，不能替代用户所装 wheel 的实测。

只改变 pair metadata 时无需额外 padding activation/weight。
若未来修改的是逻辑 K，或 TP 分片起点/终点不按 32 对齐，则需联合审查输入、权重和 scale
对应的分组边界；本补丁没有声称处理这一更大范围。

## 本地新增实现独立审查

审查对象：`afd_plugin/compat/patches/npu/mxfp8.py`、
`afd_plugin.__init__.register_dsv41_mxfp8_scale_patch`、`pyproject.toml` 的独立入口、
`tests/npu/dsv41_mxfp8_linear_probe.py` 和更新后的 `A5_NATIVE_EAGER_ZH.md`。

复制函数 `process_weights_after_loading(self, layer)` 的参数、缺省参数和无返回注解均匹配上游。
AST 对比确认：移除标记范围内唯一新增 odd-width `if` 后，完整函数与锁定源码相同，
包括 E8M0 bit view、float32 exponent 提取、block/group 检查、重复展开、weight transpose
及 wo_a/wo_b 分支。前置注释说明原因、功能、成本和移除条件；所有新增差异有 PATCH 标记。
补丁复用原模型构造、原量化与 TP 归约；没有涉及 HC/pre_mix、Engram、Attention cache 或路由精度替换。

成本是每个 odd shared down projection 增加 5120 个 uint8 scale metadata 字节，
40 层共约 0.195 MiB/rank，并在加载时增加一次 scale padding 分配/复制。
不增加逐 token 通信或归约；已有重复展开的加载峰值仍然存在。
删除条件为已锁定并验收的上游 DS 方法自行支持 odd pair packing。

独立 `afd_dsv41_mxfp8` general-plugin 入口受 `AFD_DSV41_MXFP8_SCALE_PATCH=1` 控制，
lazy import 只安装该方法补丁，不调用 `register_afd`。
锁定 vLLM `vllm/plugins/__init__.py:36–92` 的语义是：未设置 `VLLM_PLUGINS` 时发现的
所有 general plugins 均可加载；明确列表时才过滤。因此更新说明中的显式 allowlist
必须保留 `afd_dsv41_mxfp8` 并排除普通 `afd`，不能将 unset 误解为不加载插件。
该控制组应标为“原生模型 + 单项量化兼容补丁”，保留纯原生失败报告作对照。
editable 元数据刷新和入口枚举可以发现新增入口尚未安装的情况；不需要替换 FastAPI 或上游树。

## 后续验证与剩余限制

先在原 A5 容器执行小探针：K=256/288/320、N=5120、scale_alg=0/1，共 6 个真实 NPU case。
probe 使用实际 DS post-load 及其继承的 native apply，不创建模型/通信组或加载 checkpoint。
第二输入行只激活最后有效 group，数值比较及非零断言可发现丢失第九列的问题；
两个 even-group K 对照覆盖原有布局。它显式设置 scheme 字段，不覆盖 scheme 构造或 checkpoint loader。
必须记录输入/weight/scale shape、算法、错误、目标 wheel 及源码身份，6 case 通过后再重试完整 TP8。

完整模型仍需重新覆盖加载后处理、profile、KV、完整 FP4 MoE、原生路由、HBM、health、
首请求及输出 token/logprobs；不能将小探针或本静态审查升级为完整模型通过。
若下一次异常没有 prefix，最小诊断采集是在失败量化边界记录 `layer.prefix`、类名、
weight/scale dtype/shape、block_size、group_size、TP rank/size后原样重抛，
保留最早 worker traceback；不要吞错后继续加载或只截最后 API 错误。
