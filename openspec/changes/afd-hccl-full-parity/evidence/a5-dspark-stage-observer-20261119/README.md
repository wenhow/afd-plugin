# A5 DSpark：reject 已通过，Graph/U2 stage 观察器修正

输入原件保存为 `graph-input.zip`（用户 da47fead…）、`reject-input.zip`
（f21f1a45…）及 `outer-summary.json`（d3ddd2a7…）。可运行 `assess_report.py`
重建 `assessment.json`，包含输入 SHA、首次异常和原始退出状态。

## 本次设备证据

顶层 summary 的冻结包 SHA 为
`1bdf52cdfc3eb5f7f6250111e4bcb462cda9e1565910ea202c2c92466ff34d49`，
匹配 04da709 的包；runtime preflight 成功。它明确区分了两场景：

- `dspark-pd-reject-continue-ready`：process success=true、returncode=0，
  functional_verification.accepted=true，四个角色退出码全部为 0；无强制终止、
  无存活子进程。P 日志有 sender socket owner retired and joined。
  run_id 为 `0df182ec30c04214bd232911eab05b22`。因此本次 reject 场景已通过，
  不是此前失败仍未消除。ZIP 缺少详细 P/D 最终报告及内部 summary，不能从本包
  独立复算具体接受/拒绝 token 数；原始顶层验证结论及角色退出码保持可追溯。
- `dspark-pd-graph-u2-cancel-replace`：run_id
  `24d698d379344c7a8ef541279205964b`，首次异常仍是第一个有 token 的 eager
  返回后 Python 入口计数断言。观察点已是 embedding，本次实际 **1 次、6 行**，
  预期 **2 次**。同一 `native-graph-8` 的已有 host diagnostics 明确记录首层
  stage0/1 和第二层 stage0/1 的 compute_returned，随后 parent scope 返回。
  不能把 embedding 次数当作实际 stage 数量。
- Graph warmup/capture 已返回，capture 前后 KV 字节/物理页保持；replay=false，
  取消/替换未完成。D 随后仍有 `corrupted double-linked list` / exit -6，
  内部 supervisor 强制终止其他三角色。顶层 process 无存活子进程不等于内部
  正常退出；原始失败及退出码保留，堆损坏根因仍未解决。
- 四角色、两个独立 A/F pair、native TP1；Graph 场景启用 compute overlap、
  all axes 和 cross_layer_v1，constant7 target。组件使用受控参数，不能外推
  完整检查点、Engram、性能、物理重叠或其他拓扑验收。按用户要求不以 NPU
  占用状态解释本次失败。

## 上轮遗漏与本次最小修正

上轮只覆盖了 stage 内直接 embedding 的 host fixture，遗漏了原生 MRV2 在
模型执行之前准备 `inputs_embeds` 的分支。锁定的 vLLM `ced6857a`：
`GPUModelRunner.execute_model` → `DefaultModelState.prepare_inputs_embeds`
可先调用整个父批次的 embedding，再将 embeds 送入 eager 或 replay。
`run_deepseek_v41_stages` 在有 `stage.inputs_embeds` 时直接使用切片，故 embedding
既不保证每个 stage 一次，也不保证 replay 为零。原生摘录见
`native-input-contract.json`，未编辑任何上游文件。

新的观察点为受控 target 首层 `self_attn.forward(positions, hidden_states,
scaling=None)`；fixture 原签名一致。实际 eager stage 经 `_prepare_stage` →
`prepare_remote_ffn` 调用它；replay 不运行 Python target 层。hook 只在 capture
完成后安装，返回原始结果并在 finally 恢复；只读 positions.numel，无新增设备
同步、传输或 HBM 分配。不修改 native input preparation、生产执行、图缓存/key、
KV、FP32 路由、HC/pre_mix、共享状态或 Engram 所有权。

保留 U2 eager=2、U1 eager=1、replay=0 的严格检查，增加每次 stage 行数为 3
的检查；原数值/whole-pool/Ready/proposal/存储/报文/取消断言保持。新增
`last_returned_execution` 在断言前写入报告，失败时仍可查看实际次数/行数和
host records，计数/行数异常消息也带该记录。该 hook 是组件测试观察器，不是
新的生产 upstream 兼容补丁；未来有原生 stage observation API 时可移除。

## 本地验证和交付边界

- `observer-red.json`：先用 04da709 冻结 helper 运行扩展后的 host 检查，
  父批次预 embedding 分支复现计数断言；新增诊断字段亦在旧 helper 中缺失。
- `observer-green.json`：U1/U2 各覆盖预 embedding 和 stage 内 embedding；
  eager 少执行 stage、replay 执行 Python、stage 行数错误均拒绝，异常恢复 hook。
  与入口环境/CLI 检查合计 **21 项通过**。标准库 fixture 不执行 torch/NPU，
  不证明设备 stage 算子、replay、取消替换或正常退出。
- `source-verification.json`：与 04da709 包相比，仅 graph helper、provenance、
  manifest 改变；146 份生产源码、native identity 和其 194 个 pins 保持。
  186 份 Python 编译、确定性重建、Ruff、规划及上游保护见 `validation.json`。
- 初次规划检查因只读 vLLM 目录 `git status` 超过 30 秒失败，原记录保存在
  `validation-first.json`；只重试该项后通过，其他已通过检查未重复执行。
- 冻结后继仍从严格父 helper 哈希生成，替换必须唯一；完整差异见
  `observer-successor.diff`。未提交工作区其他生产/测试/文档增量。
- reject 的本次通过与 Graph 的失败分开记录；任务整项及堆损坏仍待验证，不以
  修正观察器自动宣称新包 A5 通过。不新增 A3/A5 推理、profiler 或性能场景。

## A5 重跑

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 --preflight-timeout 900 --dspark-only \
  --output-dir results/dsv41-a5-dspark-stage-observer-20261119 --run
```

保留原始两个 DSpark 场景及完整 Graph/U2 参数。回传顶层 summary 和失败场景
JSON/log 即可；普通五场景不用重跑。若新断言仍失败，报告会保留失败批次实际
stage 次数/行数，不需再从缺失的 executions 条目反推。
