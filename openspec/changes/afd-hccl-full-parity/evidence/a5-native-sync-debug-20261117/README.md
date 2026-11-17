# H=1024 后的原生异步异常与同步诊断入口

输入为用户上传的`aab351a8beb94dd0931099cc8103b0c2.zip`，只有
`native-eager-u1.log`。原始字节保存在`native-eager-u1.log.gz`，SHA与主要字段见
`failure-summary.json`。日志确认实际H=1024、Attention D512、Index32/D128，
模型加载、KV初始化和公开warmup返回；首轮没有完成任何请求步骤或采样，
worker shutdown仍记录失败。不能把进程组已销毁等同于完整清理或运行通过。

异常为`[Bisheng] Launch kernel failure! ret 507035`，随后stream 61的event record
报向量核异常；Python栈停在`torch.ops.vllm.dsa_v41_forward`外层。
运行时提示该栈可能因异步执行而不准确，并建议`ASCEND_LAUNCH_BLOCKING=1`。
本次日志没有具体内核名称或内层算子调用，不能确认是CompressorV2、SMLA、QLI
或更早提交的算子。H下限违规虽已修正，首轮设备错误仍然存在。

集中入口新增`--diagnose-native`：

- 子进程创建前设置`ASCEND_LAUNCH_BLOCKING=1`，不修改父环境或共享安装。
- 只执行既有原生eager/U1场景，保留原生预检、真实Worker、预算和清理。
  A/F、Graph/U2、多流和DSpark场景不运行。
- `summary.json`声明`diagnostic_only=true`；设备运行成功仅记为`diagnostic_only`，
  不产生原生/AFD比较，不宣称七场景通过。失败继续返回非零并保留日志。
- 原生probe同时记录实际继承的`ascend_launch_blocking`。
  默认不带参数的七场景命令保持原来的执行范围和环境继承。

检查与来源：

- Windows六项stdlib CLI检查通过：原vendor四项，加单一原生场景及真实子进程同步变量、
  父环境/普通子环境保留两项检查。metadata只输出声明，不导入NPU运行时。
- `run_cpu_cli_validation.py --output-dir .../cpu-cli-003`在既有A3解释器的全新隔离目录
  运行相同六项CLI检查，6P/0F/E；测试进程未导入torch、torch_npu、vLLM、
  vLLM-Ascend、afd_plugin或cann_ops_transformer。原始日志在`results.tar.gz`内，
  实际命令、退出码、源码身份在`summary.json`、`remote.json`、`source.tar.gz`内。
  不执行模型加载、NPU算子或采样。
- 冻结包185个Python文件编译，194项原生Git身份检查通过；146个生产模块逐字节不变。
  与`parent-bundle/`比较只有launcher、probe的观察字段及生成manifest变化。
- Ruff、格式和本地规划/上游保护检查通过；不计为A5验收。

需用户更新后按文档的`--diagnose-native --run`命令采集同步栈，回传原生日志、
summary与native-runtime身份。当前尚未定位并修复具体设备算子，不变更设备任务勾选。
