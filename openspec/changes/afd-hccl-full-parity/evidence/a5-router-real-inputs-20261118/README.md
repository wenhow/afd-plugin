# A5 路由随机对照通过与原生输入定位（2026-11-18）

用户上传的`ee5a52b081874d3ba24a851cd876b18a.zip`原样保存为
`router-diagnostic-upload.zip`，SHA256为
`5c557e701ee22b28ad451f3df8f7441d2246a312944c2e5ea909f7de36940355`。
用户明确要求忽略日志中的NPU占用提示；`operator-assessment.json`单独记录此选择，
原始summary和日志未改写。此轮没有在A5代替用户运行设备实验。

## 上传结果

六项均返回0、没有残留所属子进程，实际算子返回且CPU参考匹配：

| 形状 / dtype / top-k | 权重最大绝对误差 |
| --- | ---: |
| 8×64 / FP32 / 2 | 1.1920928955078125e-7 |
| 8×384 / FP32 / 2 | 5.960464477539063e-8 |
| 8×384 / FP32 / 6 | 5.960464477539063e-8 |
| 8×64 / BF16 / 2 | 0 |
| 8×384 / BF16 / 6 | 0 |
| 8×384 / FP32 / 6 / vision-bias | 5.960464477539063e-8 |

ID全部精确一致。没有Worker、模型或通信组执行，vision-bias只作为算子额外对照。
原始summary的`success=false`、`npu_acceptance=failed`与空闲观察为false一同保留，
六个子结果均通过且父summary没有异常栈。占用观察不参与本次算子判断，
也不能证明上一份64专家原生Worker的507035/341故障已消除。

六项实际加载库路径及SHA一致，详见逐项assessment。主要身份：

- `/opt/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so`：
  `c0476a67006e7eeea2b34f17ff103be8034852aeb4b07a923a0e6436002a7da6`。
- Ascend安装内的custom_transformer `libcust_opapi.so`：
  `e896e0387f6ddad9e8088ca3aa636e414da31f31607d9fa78d4ef090c9df1595`。
- 外部CANN custom_transformer `libcust_opapi.so`：
  `e8861d0e94710c8cc82dba89f99877eeee78fa2e0aeffa9512085d3df1ffe056`。

原生预检194份Python源码匹配；包为torch2.10.0、torch-npu2.10.0.post4、
vLLM0.30.0+empty、vLLM-Ascend0.19.1rc2.dev637+g2db8549c5。
Python匹配及这些库SHA仍不证明C++二进制的构建源码。

## 实现与验证

`--diagnose-native`以线程局部的Python调用观察器保存原生路由实际输入、schema、
布局、有限值统计及已加载库身份，原样调用原生算子。失败后在所属进程回收的前提下，
使用新进程重放完整、全部有限的失败调用；逻辑值/形状/步长/偏移保持，
每张量存储限制1,048,576个元素。未向上游类或函数赋值，未修改路由数学或模型参数。
重放仅检查算子返回、输出有限及ID范围，`reference_matched`保持false，
未实现实际输入的完整数值等价验收。原生失败在父报告中始终保持失败。

同步读回会改变诊断时序；若读回就失败，不能直接归为路由错误。
重放不能复原完整模型上下文或底层存储的其他内容；输入非有限或记录不完整时明确跳过。
随机对照通过使继续调整专家数缺少充分依据，本轮保持H1024、E64、top-k2和Index32/D128。
新增`--ignore-npu-occupancy`仅用于诊断，保留占用观察和所属子进程清理要求，
默认七场景不能使用该开关。

`cpu-profile-cli/`在A3隔离目录运行15项真实CLI/子进程metadata和报告接缝检查，
0失败/错误，未导入torch、torch_npu、vLLM、vLLM-Ascend或afd_plugin；本地同15项通过。
未新增tests/unit的harness测试。
一次完整包SSH传输停滞，尚未启动测试；本目录`run_cpu_cli_validation.py`改用128KiB分块传输后，
在新隔离目录完成最终源码检查，没有修改共享环境。
`cpu-profile-capture-final/`由本目录`run_cpu_capture_validation.py`执行，真实CPU PyTorch
2.10.0+cpu验证输入/输出保持、原异常和输入保留、NaN/Inf统计、非连续布局及非零偏移恢复，
6项通过、0失败/错误，NPU未初始化，传输源码前后SHA保持。
另验证opaque外层算子内部的路由调用可见，以及异常后原线程profiler恢复。
锁定调用链为`vllm::moe_forward_shared` → `AscendMoERunner._forward_impl` → 原生router。
CPU实测普通TorchDispatchMode只见外层，故使用作用域内的`sys.setprofile`观察真实Python op调用，
退出时恢复原线程观察器；不改变dispatcher注册或原生kernel。
这些CPU检查不执行或模拟A5设备算子，不作为设备正确性证据。

准备脚本编译186份Python、核对194份锁定Git对象；Ruff和本地规划/上游保全检查通过。
冻结父包保存在`parent-bundle/`；与父包相比只变更三个诊断脚本及bundle-manifest，
146份生产模块逐字节相同。当前包SHA与launcher SHA见默认`bundle-lock.json`。
未夹带本地其他MTP/Engram改动，未修改vLLM、vLLM-Ascend或donor工作区。

下一步按[集中指引](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)运行A5原生输入诊断，
上传完整目录（含输入PT文件）。实际输入重放、原生首轮、七场景、完整checkpoint/Engram及性能
仍未验收，不改变任务勾选。
