# A5 dSpark compressed MX 兼容修复包

该包修复固定 vLLM-Ascend `3da28f941` 无法识别 DeepSeek-V4-Flash-DSpark checkpoint 中 `W8A8 MXFP8` 和 `W4A8 MXFP` compressed-tensors 配置的问题。

安装不会重装 Python/CANN/HCCL，不修改模型权重，不覆盖现有 afd-plugin 目录。包内只携带两个代码 diff，不再携带完整 Git bundle。安装器在干净的 vLLM-Ascend 基线上重建固定修复提交，并从现有 `66ec72f` A5 dSpark 工作树创建新的本地 worktree 后应用 afd-plugin 增量补丁。

## 安装

```bash
tar -xzf dsv4-a5-dspark-compressed-mxfp-patch-20260917-r2.tar.gz
cd dsv4-a5-dspark-compressed-mxfp-patch-20260917-r2

# 仅当实际路径不同才修改。
vi config.env
bash install.sh install 2>&1 | tee install.log
bash install.sh check
```

安装器要求：

- vLLM-Ascend HEAD 是包内记录的基线或目标提交，且工作树干净；
- afd-plugin 源目录是此前安装的 clean `66ec72f` 工作树，或已应用“指导书路径修正”补丁、tree 与 `1720b71` 相同的工作树；现场通过 `git am` 生成的提交哈希可以不同，例如此前采集到的 `90f39082`；
- 新 afd-plugin 目标目录不存在，或已是包内固定目标 tree 且工作树干净；
- checkpoint 的 `config.json` 和 `model.safetensors.index.json` SHA256 与现场已收集值一致；
- 旧 vLLM-Ascend checkout 中已经存在编译好的 custom ops。

安装器只对 afd-plugin 执行 `pip install --editable`，用于把现有 venv 指向新工作树；不会安装或升级依赖。设置 `SKIP_EDITABLE_INSTALL=1` 可跳过这一步。

安装成功后使用输出中的 `AFD_PLUGIN_ROOT`、`DSV4_VLLM_ASCEND_ROOT` 和 `MODEL_PATH`。其余 A5 环境变量及验证步骤见 `DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md`。

先重跑 no-AFD 门禁：

```bash
cd "$AFD_PLUGIN_ROOT"
export PHASE1_NATIVE_OUTPUT_ROOT="/data/validation/dsv4-phase1-a5-dspark-mxfp-$(date +%Y%m%d_%H%M%S)/native-dp4-dspark"
bash tools/dsv4/run_phase1_a5_native_smoke.sh run-dspark
```

失败时保留整个 `PHASE1_NATIVE_OUTPUT_ROOT`；不要修改 checkpoint 配置或屏蔽 clean-tree 审计。
