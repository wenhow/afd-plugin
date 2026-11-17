# A5 正常场景配置初始化检查：2026-11-17

[用户回传原包](a5-results.tar.gz) SHA256：
`03fb90b3996d6297b271a47047495b2eccb7de3fadae886344baad295bcab52c`。
原生运行时预检194份源码匹配，40.429554秒、退出0、无超时/强杀/存活进程；
解除遗留锁后已越过先前的导入阻塞。接着第一个`native-eager-u1`场景21.816667秒退出1，
同样无超时/强杀/残留。

原始`native-eager-u1/result.json`中配置已完成且几何正确，NPU初始化由False变为True，
随后在验证probe的`npu_initialized_after_config is False`断言失败。
`model_loaded=false`、`inference_executed=false`、`afd_modules_imported=[]`；
没有执行模型或A/F比较，整轮保留失败。

## 修复范围

`dsv41_functional_worker_probe.build_config`将实际执行与独立CPU配置检查分开：
`--run`在原生配置查询前绑定`args.device`，记录配置前后实际可见设备序号并验证相同，
保留真实初始化状态，再走原有Worker factory/init_device/load/KV/warmup/采样流程。
`--run-config`继续要求配置前后NPU未初始化。

锁定vLLM/Ascend的配置路径可查询设备信息，并未承诺该过程不得初始化设备。
例如vLLM`ced6857a:vllm/engine/arg_utils.py:get_batch_defaults`调用平台设备信息接口；
Ascend`7e37e29a:vllm_ascend/platform.py:get_device_name`调用`torch.npu.get_device_name`。
这说明旧断言超出了配置接口契约；具体触发本次初始化的内部调用尚未采栈定位。
原生Worker仍以同一个`args.device`作为`local_rank`进入原生设备绑定，未改上游源码或替换算子。

[前一版冻结包及锁](parent-bundle/bundle-lock.json)完整保留，父归档SHA为
`3d6da6143d19a72886162c260b01257caa00f7643b99c9d356d1b98dddfffe9e`。
新默认归档SHA为
`43c0f8b20e473722401ad3037ccaf1f2e1679caa32e9c09f7ada44b659dee759`。
188个源码成员仅functional worker probe变化；146生产模块、194份原生身份锁与入口字节不变。

## 本地验证与待验

[CPU报告接缝检查](cpu-reporting-validation.json)用受控Torch/NPU与配置返回值重放报告行为：
旧probe拒绝所记录的A5初始化状态；新probe允许实际执行初始化并保留0号/非零设备；
CPU配置路径仍拒绝初始化、接受纯host；配置改变所选设备仍拒绝。六项符合预期。
这些是合成CPU检查，不构成设备绑定、配置查询或模型执行的NPU验收。

[七场景metadata](metadata-validation.json)退出0且不导入runtime；185份Python编译、
194个锁定原生Git对象核对、Ruff检查及格式检查通过。
实际A5配置后继续Worker/加载/推理和完整七场景仍待重跑，任务不提前勾选。
