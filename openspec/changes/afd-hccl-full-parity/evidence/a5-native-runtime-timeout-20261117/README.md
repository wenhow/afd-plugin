# A5 原生运行时预检超时：2026-11-17

本次回传确认原生源码身份已匹配，失败是外层 120 秒预检超时终止。
修复仅调整验证入口的时限与错误说明，不构成 A5 导入、算子或七场景验收通过。

## 原始证据

- [用户回传报告](a5-preflight-reports.zip) 包含 `native-runtime.json` 与
  `native-runtime.process.json`；原ZIP SHA256为
  `eff910d8d5f8231b9453f3f7a87edafcd6a3579507447ebcd66bb060d88280ec`。
- [原始运行时日志](native-runtime.log.gz)解压后的SHA256为
  `d6515800af3080967397d0901323796c20436cecabac4b9fe6dfcc58ac125e88`，
  记录 Ascend 初始化与多个 `cann_ops_transformer` JIT 编译/链接输出。
- 原生 `sources_match=true`、`source_count=194`；报告仍为初始 `status=failed`，
  没有最终异常栈。进程记录为 `timed_out=true`、`returncode=-9`，强制清理
  PID 89233/89232/89231/88864；清理采样仍有 `surviving_owned_pids=[89233]`。
  这只能说明当时仍观测到该进程，不能推断其当前状态，原失败与清理问题保留。
- 该日志中的 `minimax_sparse_attention_split_kv` 提示后仍有后续编译输出，
  不据此认定其为终止原因。七个功能场景尚未开始。

## 修复与源码身份

统一入口新增 `--preflight-timeout`，默认600秒、必须为正数，独立于单场景
`--timeout`。所用值写入 `summary.json` 的 `preflight_timeout_seconds`。
超时错误明确说明时限和诊断文件，其他预检失败保留退出码和原始报告路径。
原有失败判定、强制清理和存活进程检查保留。

[原冻结包及锁](original-bundle/bundle-lock.json)与原准备记录完整保留在
`original-bundle/`，其归档SHA256为
`87391afb4559f73a07ddf2211f17e31d8dba40b3d0e6efbaeeb7bb01f7ca51c0`。
仓库默认交付包同步更新为
`3d6da6143d19a72886162c260b01257caa00f7643b99c9d356d1b98dddfffe9e`；
新入口SHA256为
`c05e717ebf3064acc6e148b8c602371b77d29bc9bc9e4c8418af58c8141b9d14`。
188份源码中仅入口一份发生变化，146生产模块和原生194份源码身份锁逐字节不变。

## 本地检查与待验

[检查记录](local-validation.json)包含默认600秒及显式900秒的两次真实七场景metadata运行，
均退出0且`runtime_imported=false`、NPU验收pending；三个非法参数CLI均退出2，未创建输出。
CPU报告接缝重放分别输入本次真实超时记录及非超时失败记录，均退出1、零场景，
确认错误说明可区分两类失败，并保留原记录的存活PID。重放中的进程/NPU状态是受控输入，
只证明错误报告行为，不是设备或真实超时执行证据。
归档成员校验、185份Python编译和194个锁定原生Git对象核对通过。

重跑见[集中验证指导](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)。
入口和归档/锁/准备记录必须一起更新；保留旧结果并换新输出目录。
首次JIT总耗时、600秒是否足够、清理以及七场景功能仍待A5实际重跑，任务不提前勾选。
