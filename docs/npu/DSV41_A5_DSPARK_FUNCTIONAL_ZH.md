# A5 DSpark 两场景功能验证

更新仓库后，在已配置好的 A5 Python/CANN/ATB/OPP 环境执行一条命令。入口依次运行两个四角色组件，结果写入新目录。**本包尚未在 A5 执行，设备验收为 pending。**

```bash
/path/to/venv/bin/python -B tests/npu/dsv41_dspark_a5_validation.py \
  --model /path/to/DeepSeek-V4.1-Flash \
  --python /path/to/venv/bin/python \
  --devices 0,1,2,3 \
  --timeout 360 \
  --output-dir /path/to/results/a5-dspark-functional-001 \
  --run
```

`--devices` 为当前进程可见的四个不同编号，顺序为 P、D、F1、F2；现有 `ASCEND_RT_VISIBLE_DEVICES` 映射保持不变。输出目录必须不存在。每场景期限须小于源页保留时间 720 秒，外层额外保留 60 秒收尾。当前入口沿用八卡环境，开始、场景结束及最终 `npu-smi info` 均须报告八卡空闲。

不加 `--run` 可在普通 CPU 主机执行源码校验和两个原入口的 metadata 分支，不导入 Torch/vLLM，不查询 NPU，不要求模型目录存在：

```bash
python -B tests/npu/dsv41_dspark_a5_validation.py \
  --model /readonly/model \
  --output-dir /fresh/path/a5-dspark-dryrun-001
```

## 两个场景

1. **Eager/U1：Ready 与拒绝后继续。** P 的真实 native ExecuteModelState/aux 经原生 proposal、projection consumer 和 typed Ready/read/DONE；D 原生 Scheduler/execute/sample/propose 连续三轮。受控 target 预期输出 `[7]`、`[7,8]`、`[7,8]`，computed 前沿为 5、7、9，累计五个输出 token 后自然长度停止。保留 proposal 前后的精确 not-ready 负查询与下游唯一真实正 lease。
2. **Graph/U2：全接受、取消及新请求。** P 顺序生成三个真实 source prefix 并保留至各自 DONE。D 同一 owner/Scheduler/receiver/TE 运行 A/B 两轮，取消 A，B 继续至长度停止，真实空步退休后接入 C。C 首轮 eager、下一轮 U1 replay，再正常退休；C 的实际请求槽须来自已退役 A/B。物理块交集仅记录，不保证固定 ID。五轮 phase 为 eager/replay/replay/eager/replay；valid 行与 padded 行分别保存。

四个角色必须全部退出 0，原 supervisor 必须成功且没有超时或强杀。角色写出 `success:true` 后 SIGABRT 仍使整轮失败；前场景失败即停止后续，保留原日志与 JSON。

## A5 必要差异与原生边界

源码包来自冻结的 `engine-args-pd-green-001`（447 源，SHA `af0c18647dc8f25bb42a901ae40a74bb0f796f75ceef23dc6259ca50964cebdd`），**146 个生产模块逐字节保留**，不读取当前工作区其他生产改动。原入口可达的 29 个 donor helper 中，25 个保持原字节，4 个具有明确 A5 后继差异，完整前后 SHA 与 diff 见包内 `a5-helper-delta.json`：

- Speculator 配置在首次构造前声明 64 个 Q heads，H512/head512/HC4、q_lora_rank64 保持；不在绑定后修改 HF，不修改原生 metadata 的 64-head ABI。
- Decode worker 使用已安装 A5 原生扩展注册，独立 CPU oracle 按实际 typed cache 解码、量化及写入完整物理页。
- 受控 target Attention 对 A5 原生 uint8 SWA 使用原 `MixedQuantPackedCacheOps.write_attention_cache(kind='win')`，不改变原生 cache spec 或硬件 profile。
- P 首个 target 执行增加独立整池 byte oracle，随后仍执行原 sample/propose/consumer/Ready。

新增的 A5 helper 解释真实 packed 行：512 个 E4M3FN 数据字节，加 16 个连续小端 BF16 scale，共 544 字节。标量 oracle 按锁定原生 C++ 的 FP32 amax/epsilon、FP32 乘 `1/448`、scale 向上取二次幂、FP8 ties-to-even 推导；未选中范围与整个物理池继续逐字节比较。CPU 向量只能验证参考公式，实际原生 writer 与其比较仍待 A5。

A5 原生策略下 `enable_custom_op()` 返回 **False**。入口保留此结果，不强开 capability，不替换 dispatcher。设置真实设备后，沿原生平台已有范式导入已安装 `vllm_ascend_C`，核对 `_C_ascend::inplace_partial_rotary_mul` 的实际设备注册；HC 与 packed writer 仍使用 A5 已安装 `custom_ops` / `cann_ops_transformer` 的原实现，并记录实际 schema、模块和动态库身份。缺失时保留原始失败，不编译、不装包、不替换算子。

运行前后校验 **194 个锁定 Git Python 源文件**（原 190 加 A5 硬件/打包 Attention 的 4 项）。原 A3 第 191 项 Torch-NPU profiler 不作为 A5 期望。Torch/Torch-NPU、Mooncake single-owner extension、SoC、原生扩展及实际已加载 CANN/ATB/OPP/wheel 库仅记录 A5 当前路径与 SHA；不递归扫描整个 OPP 树，不携带 A3 `.so`、vendor、CANN 目录或 JIT cache。

这是**受控 target/FFN + 真实 native draft**组件。原 draft Attention/HC/MoE、proposal、共享 embedding/head、拒绝采样及 PD 保持原实现。受控 target 算术与 logits policy 不代表 checkpoint 数值等价；Markov 两参数在首次执行前用正常 weight loader 声明，原 native logits/propose 不变，权重字节与指针均核验。`--model` 是只读配置/tokenizer 来源，输出目录中另建受控配置，不加载完整 checkpoint target 权重。

本轮不验收完整 Worker/服务、完整模型精度、Engram、性能或物理流重叠。Graph 使用 stage_serial，五个 overlap 轴关闭，draft 保持 eager。最初仅通过 CPU dryrun 的候选已因 A5 bootstrap/packed ABI 两项源问题被替换，不能视作 A5 成功证据。

## 结果和源码重建

顶层 `summary.json` 给出整体结果；场景日志、`.process.json` 和原场景目录保留角色报告、控制协议、输入/输出/history、FP32/BF16 oracle、整池 KV 和权重稳定性证据。`snapshot/` 是本次实际冻结源码，`native-before.json` / `native-after.json` 为 A5 身份，`npu-*.log` 为原始空闲检查。失败时请保留整个目录。

维护者可重建同一纯源码包（需要本机历史父包及锁定 Git 对象；A5 执行无需这些历史目录）：

```bash
python -B tools/diagnostics/prepare_dsv41_a5_dspark_bundle.py \
  --output-dir /fresh/path/a5-dspark-source-bundle
```

仓库默认使用 [纯源码包](../../reports/dsv41-a5-dspark-functional/runtime-source.tar.gz) 及同目录 lock；也可用 `--source-bundle /path/to/runtime-source.tar.gz` 指定同目录有 lock 的副本。所有本机检查与源码准备记录仅是静态/CPU 证据，A5 两场景仍待实际执行。
