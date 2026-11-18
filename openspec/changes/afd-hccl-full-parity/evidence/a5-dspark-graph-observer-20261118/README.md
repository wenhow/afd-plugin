# A5 DSpark Graph/U2：首个 eager 批次观察器误判

输入是用户提供的 `77678836844e4d698a84162326de406a.zip`，原件保存为
`graph-u2-cancel-input.zip`，哈希、原始退出码和分析在 `assessment.json`。
它只包含一个失败场景：无顶层预检/冻结源码身份，也无 reject 场景成功原始报告。
reject 成功仅按用户本次反馈记录，不把不同场景合并成完整七项验收。

## 首次失败与尚未解决的异常

- run_id `2961f4d3e38240e9ad96aeb2ae8e705f`，四角色、两个独立 A/F pair，
  native TP1；Graph/U2、cancel/replace、compute overlap、all axes、
  `cross_layer_v1` 开启，target logits policy 为受控 `constant7`。
- warmup/capture 返回成功；capture 前后 KV 字节及物理页保持。P/D 初始 KV
  传输的 whole-pool oracle 已返回，但不能推出后续 decode/replay 成功。
- `kv_consumer.json` 首个异常在 graph probe `observed_execute` 的
  `assert entry["python_target_calls"] == python_calls`。第一个有 token 的 eager
  parent 已返回，预期两个 Python target 调用，顶层 `model.forward` 记录为零。
  验证尚未进入 replay、sample、取消或替换完成阶段。
- 后续 D 日志仍有 `corrupted double-linked list`，D exit=-6，FFN 对端断开，
  supervisor 强制终止其他三个 PID；非 timeout。时间顺序不足以证明断言导致
  堆损坏。两者分开保留，不能把观察器修正写成堆损坏已经消除。
- controlled target/FFN 参数不等于完整训练检查点；full_model_accepted=false。

## 调用链与最小修正审查

`HcclV41UBatchRunner.run` → `run_staged` → `execute_hccl_v41_stages` →
`run_deepseek_v41_stages`。该路径直接执行 stage，不调用模型 wrapper 的 forward。
后者不能作为 U2 eager 是否执行的依据。每个非空 stage 使用原始 token IDs 调用
`owner.model.language_model.model.embed_input_ids`；U1 的 staged 路径也使用它。

只把测试观察点移到这个共同入口，并记录 `python_target_entrypoint` 和每次
`token_rows`。U2 eager 仍必须两次，U1 eager 一次，replay 零次。原有 host phase、
retained graph storage、hidden/aux 指针、numerical、whole-pool、proposal、
transport 及取消生命周期断言保持。hook 只在 capture 返回后安装，在 finally
恢复；不替换 embedding 输出，不更改调度、图选择或设备计算。

修改由 `prepare_dsv41_a5_functional_bundle.py` 对已冻结 helper 的精确父哈希
生成，替换点必须唯一。`observer-successor.diff` 是完整 helper 差异；未把工作区
其他未交付 helper 纳入。生产及上游源码零修改，因而不需要新的生产兼容 patch。
保留这个 host observer，直到统一组件入口提供可替代的 stage 执行证据。

## 本地验证

- `host-observer-red.json`：旧冻结 helper 的 U1/U2 staged 正常路径均被旧计数
  错误拒绝，复现同一断言。输入是标准库 host fixture，没有 torch 或 NPU。
- `host-observer-green.json`：新冻结 helper 通过 U1/U2 eager→replay 检查；
  缺失 eager stage、少一个 stage、replay 进入 Python 均被拒绝；异常退出恢复
  hook。结合既有 CLI/环境检查共 **21 项通过**。这些检查只证明 host 观察器契约。
- `source-verification.json`：与 09111d9 冻结包相比，仅 graph helper、provenance、
  manifest 改动；146 份生产文件全部字节相同；保留 194 个原生 source pin。
- 构建编译 186 份 Python；Ruff、本地规划/上游保护和确定性重建见
  `validation.json`。不运行新的 A3/A5 推理；设备结果待用户回传。
- 同步修正上轮功能指南及 tasks 新增段落被 shell 编码替换为问号的问题；保留
  这些文件之前已有的本地修改。无关修改未提交。

## A5 重跑

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 --preflight-timeout 900 --dspark-only \
  --output-dir results/dsv41-a5-dspark-graph-observer-20261118 --run
```

保留两个原始 DSpark 场景及完整 Graph/U2 参数，跳过普通五场景。回传顶层
summary/预检/源码身份 JSON 和失败场景的 JSON/log 即可。Graph/U2 replay、
取消替换及四角色正常退出必须由这次实际运行证明，整项验收仍待确认。
