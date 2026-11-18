# A5 真实输入重放与算子导入时序（2026-11-18）

两份用户上传ZIP以原文件名原样保留，SHA及结构化判断见`assessment.json`。
其中native输入和重放结果同时单列在`native-inputs/`，输入PT的SHA为
`6bf7ca17a89d2e6dbcbc1c9878b567a5e2a3dca2357d1be2a84eaad37e93d6e0`。
按用户要求排除NPU占用提示；原生失败、Worker shutdown失败及原始summary均不改写。

## 已观察到的结果

- 原生H1024/E64/top-k2/Index32-D128、eager/U1首轮仍报507035/341 UB越界；未完成采样。
  所属进程最终回收，未超时或强杀；原生分布式组销毁完成。
- 真实logits为FP32 `[8,64]`、stride `[64,1]`、offset0，有限值范围约`[-3.82e-5,2.91e-5]`；
  bias为FP32 `[64]`、stride1、offset0，范围约`[-9.92e-4,9.76e-4]`。两者均无NaN/Inf。
- 同一PT输入和标量参数在全新A5子进程重放成功，输出有限且ID范围合法；没有模型或通信组执行。
  原始重放报告的`reference_matched=false`仍保留；本轮另用真实CPU PyTorch计算参考，
  ID精确一致，权重最大绝对误差`5.960464477539063e-8`。
- 两进程共有的opapi/扩展库路径和SHA一致；原生另外加载了
  `/usr/local/Ascend/cann-9.2.0/opp/vendors/customize/op_api/lib/libcust_opapi.so`，SHA为
  `c6592b7bafefdfb1f8333f5878582e371205f1596c51113e09ec606e5350b296`。
  仅凭已加载库列表不能判定它是否提供了失败路由符号。

因此，不能把这次故障确定为输入非有限、小专家形状或普通stride问题；运行时上下文、
实际符号绑定、库加载/缓存时序仍需要区分。未确定设备根因，也未证明启动已修复。

## 源码依据和修改

只读检查vLLM-Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`：

- `csrc/aclnn_torch_adapter/op_api_common.h:257`附近的`g_custom_lib_path`及
  `g_default_custom_lib_path`在共享库导入时构造，记录当时的环境路径和默认vendor优先级。
- 同文件`GetOpApiFuncAddr`先依次搜索上述表，再查transformer/common opapi；
  `EXEC_NPU_CMD`另外保存各API的首次解析地址。
- `vllm_ascend/utils.py:455`的原生bootstrap前置安装目录，目录已经存在于列表时不重新排序。
  当前普通Worker的A5 bootstrap在配置、Worker和设备初始化之后执行。

本轮在两个A5探针的实际`--run`路径中，先以stdlib查找已安装vLLM-Ascend的位置，
将其`_cann_ops_custom/vendors/custom_transformer`前置、去除该目录自身重复项，
再导入torch/vLLM等运行时。外部Transformer及其他根目录和LD_LIBRARY_PATH均保留。
未改变进程外环境、CANN配置或上游文件；既有`--run-config`及metadata不执行这一准备。
记录`operator_environment_preparation`的before/after、库SHA和`extension_already_imported`，
以便检验准备是否发生在扩展导入前。该调整对普通Worker和独立路由探针一致应用。

首个原生路由调用和重放调用前额外记录库及符号表、inference/grad/deterministic标志。
ELF探针以`RTLD_NOLOAD`取得已加载库的handle，用dlsym/dladdr查询两个路由API的导出来源，
然后平衡关闭handle；不执行任何路由API，不加载新算子库，也不证明C++缓存的实际绑定。
此处只收集上下文，没有切换原生推理模式、数学或精度，也没有替换原生算子或绕开失败。
146份冻结生产模块保持原样，本轮是A5验证入口的加载时序调整。

## CPU和静态验证

`cpu-cli-final/`：本地/A3隔离目录各17项CLI、metadata、路径优先级、缺库保留和子进程检查，
0失败/错误。新测试的包入口主动抛异常，确认查找包位置时没有执行其运行时代码；
安装目录从原列表第二位前置后，其他条目及LD_LIBRARY_PATH保持。测试不加载占位SO。

`cpu-capture-final/`：真实CPU PyTorch 2.10.0+cpu的7项捕获、异常、布局、嵌套调用、
线程观察器恢复、已加载libc符号查询检查通过；另有上传真实输入对A5重放结果的CPU数值参考。
NPU未初始化，传输源码前后SHA保持，没有启动新的A3设备推理。
首轮metadata检查在_ctypes导入时报符号缺失，改为仅在实际ELF检查时延迟导入，
最终metadata检查无需加载_ctypes或任何NPU运行时。

默认包编译186份Python、核对194份锁定Git源；与`parent-bundle/`相比只更新两个诊断脚本及清单，
146份生产字节保持。Ruff及本地规划/上游保全检查通过，详见`source-verification.json`。
所有源码和复跑控制程序在仓库内；临时目录仅为产物。

按[集中指引](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)在A5重跑原生诊断，
使用全新目录并回传全部内容。安装目录优先级修正的设备效果、原生首轮和七场景仍待验收；
任务勾选不变，不据此验收完整模型、Engram、DSpark组合或性能。
