# A5 模型加载后的参数读取失败：2026-11-17

[用户回传日志](native-eager-u1.log.gz)保留原始字节，解压后24,516字节，SHA256：
`affb5d31bbfde7a54ad9863035e830ba531a5c604c64af61e2e777ac7c98b61d`。
日志来自 `dsv41-a5-functional-20261117-rerun2/native-eager-u1`。

原生Worker初始化与模型加载已完成，日志记录加载1.08 GiB、耗时1.374674秒；
`model_loaded=true`。随后验证脚本在 `record_native_model -> parameter_record`
对参数执行 `detach().cpu()`，触发 `copy_ do not support internal format`。
`attention_kv_initialized=false`、`inference_executed=false`，尚未进入KV/推理或A/F比较。
本次Worker报告 `cleanup_failures=[]`、`native_groups_destroyed=true`；
仅有这一份角色日志，不能据此判断外层进程或所有设备的最终状态。

## 修复与接口依据

共享 `dsv41_normal_model_probe.parameter_record` 对NPU参数的detached观测张量检查格式，
仅在非ND时调用非原地 `torch_npu.npu_format_cast(..., torch_npu.Format.ND)`，
再复制到CPU、按逻辑连续元素计算SHA/finite。原参数的设备、指针、stride和offset仍取原件；
不修改模型参数、全局格式开关或上游算子，不跳过参数对照。
SHA描述逻辑元素字节，不包含私有存储的padding；格式转换会临时分配读取用张量，
属于非计时观测，不能用于性能结论。

[官方格式转换接口](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/%28beta%29torch_npu-npu_format_cast.md)
声明ND可由 `2` 或 `torch_npu.Format.ND` 指定。
锁定Ascend `7e37e29a` 的 `quantization/methods/w4a8/w4a8_mxfp4.py`
也在 `restore_weights_for_rl_loading` 使用 `npu_format_cast(..., 2)` 恢复ND；
本probe只借用转换接口，不执行该加载恢复方法或改写parameter.data。
这些说明接口选择；本地未执行A5转换内核。

准备器显式刷新共享model helper，避免它继续继承父包中的旧字节。
[父冻结包及锁](parent-bundle/bundle-lock.json)完整保留，归档SHA为
`43c0f8b20e473722401ad3037ccaf1f2e1679caa32e9c09f7ada44b659dee759`。
新归档SHA为
`817d04ee81dc55a1e3d9053dfdd424eeb83df8f1045f09367dec4359d8d320f7`。
188个源码成员仅该helper变化，另外更新manifest；146生产模块、194原生身份锁和入口均未变。

## 本地验证与待验

[可复跑CPU检查](validate_cpu_readback.py)及[结果](cpu-readback-validation.json)
使用真实CPU Torch，核对scalar、带offset的非连续view、BF16、整数、非有限值和空张量的
逻辑字节SHA/finite，并确认原值、指针、stride、offset与requires_grad保持。
六项通过，CPU路径未导入torch_npu；不模拟或验收设备格式转换。

[七场景metadata结果](metadata-validation.json)退出0且所有场景 `runtime_imported=false`；
185份Python编译、194原生Git对象核对、Ruff和格式检查通过。
[规划检查](harness-check.json)与[上游保护检查](harness-guard.json)通过。
实际A5格式转换、KV、推理、原生/AFD比较及七场景设备验证仍待使用新目录重跑，
旧失败记录与任务勾选保持原状。
