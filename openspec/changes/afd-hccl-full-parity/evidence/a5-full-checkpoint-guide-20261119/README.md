# A5 完整权重指导和诊断入口：本地交付证据

用户要求继续并给出完整训练权重验证指导。本轮入口见
[完整权重操作](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)。
用户在 A5 手工执行；本轮本机没有启动服务、读取权重 payload 或初始化 NPU。

## 改动及审查

- `preflight_dsv41_a5_afd.py` 的固定 checkpoint config/index、48 shard 结构校验、
  Python 3.11、插件来源和源码干净条件保持。21 份原生源码期望对齐当前锁，
  其中只有量化 w4a8_mxfp4、DSpark speculator、ascend_forward_context 和
  fused_experts 四份相对旧 4ec6dc77 改变。报告版本同步为 ced6857a/7e37e29a，
  没有接受两套哈希或吞掉源码不匹配。
- `verify_native_pins.py` 只读锁定仓库的 Git 对象。`native-pins-red.json` 保留
  原预检的四处失败，`native-pins-green.json` 的 21 份全部匹配并核对版本标签。
- `run_dsv41_a5_afd.py` 继续使用真实 safetensors、A4/F4 TP4、eager/U1、
  A 侧原生 CPU Engram；启动参数、vendor 继承、算子先验及 STOP/清理条件保持。
- 新增 completion 响应比较，保存原始参考和 `token-comparison.json`，检查输入
  token、32 个输出及首个分歧。无参考为 not_checked；缺失 prompt、null/list/截断
  JSON 等无效参考失败；实际差异先落盘再保持失败，并在最终 summary 保留。
  即使 tokens_match，仍不将历史不同拓扑/源码的单请求比较宣称为模型等价。
- 没有修改生产代码、上游、资格准入或 Engram 专项。该处是诊断工具，未复制、
  包装或 monkey-patch 上游函数。原七场景冻结包字节保持。

## 验证

新增标准库回归先在旧入口失败：5 个测试，包含 pin 不匹配、未记录比较、
输入不同和缺失参考输入未拒绝（含子用例，3 failures/4 errors）。
实现后另补 null/list/截断参考的拒绝，共 6 个新测试。
它们检查 HTTP 响应处理和落盘报告，不模拟模型或设备计算。

最终标准库回归连同原组件入口/Graph observer 共 31 项；具体命令和结果见
`validation.json`，源码及只读保护见 `source-verification.json` 和 harness 结果。
源身份检查本身不证明原生量化算子、真实载权、HCCL、请求或清理已经在新版本通过。

## 仍待 A5

完整权重预检、A4/F4 载权、固定 32-token 请求和正常退出；有原生参考时核对输入/
输出。此前真实权重首 token 差异继续 pending。当前脚本仅一个短请求，不自动
测试连续请求、logits、完整模型多种模式或性能，hardware_acceptance 和
model_equivalence_accepted 不自动转 true。

历史 4ec6dc77 的原生/AFD 结果和失败 summary 保持，不追认为本次 7e37e29a 验收。
新指导给出只预检和显式 --run 两步，无需重复此前七个组件场景。
