# A5 dSpark 官方上游基线恢复包

该包把 A5 验证恢复到未修改的固定上游：vLLM `0fc695fc`、vLLM-Ascend `3da28f941`。此前实验性 vLLM-Ascend `18a0709c` 若已安装，安装器只在其工作树 clean 时将 checkout 切回父提交 `3da28f941`；包内不包含、也不会应用任何 vLLM 或 vLLM-Ascend 代码补丁。

安装不会重装 Python/CANN/HCCL，不修改模型权重，不覆盖现有 afd-plugin 目录。包内只携带 afd-plugin 增量 diff，不携带完整 Git bundle 或任何上游源码。安装器从现有 clean afd-plugin 工作树创建新的本地 worktree，然后只应用 afd-plugin 补丁。

## 安装

```bash
tar -xzf dsv4-a5-dspark-official-upstream-patch-20260917-r3.tar.gz
cd dsv4-a5-dspark-official-upstream-patch-20260917-r3

# 仅当实际路径不同才修改。
vi config.env
bash install.sh install 2>&1 | tee install.log
bash install.sh check
```

安装器要求：

- vLLM HEAD 必须是 clean `0fc695fc`；安装器只审计，不修改；
- vLLM-Ascend 必须是 clean `3da28f941`，或此前由本交付产生的 clean `18a0709c`；后一种状态会被安全切回 `3da28f941`；
- afd-plugin 源目录可为此前 r2 创建的 clean `afd-plugin-phase1-a5-dspark-mxfp`，也兼容较早的 `66ec72f`/指导书修正工作树；
- 新 afd-plugin 目标目录不存在，或已是包内固定目标 tree 且工作树干净；
- 旧 vLLM-Ascend checkout 中已经存在编译好的 custom ops。

安装器只对 afd-plugin 执行 `pip install --editable`，用于把现有 venv 指向新工作树；不会安装或升级依赖。设置 `SKIP_EDITABLE_INSTALL=1` 可跳过这一步。

安装成功后使用输出中的 `AFD_PLUGIN_ROOT`、`DSV4_VLLM_ROOT` 和 `DSV4_VLLM_ASCEND_ROOT`。模型必须使用来源可信且未经手工改写的 DSpark 配置；若原模型目录的 `config.json` 被修改，按指导书创建软链接模型视图，不修改或复制权重。其余 A5 环境变量及验证步骤见 `DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md`。

先重跑 no-AFD 门禁：

```bash
cd "$AFD_PLUGIN_ROOT"
export PHASE1_NATIVE_OUTPUT_ROOT="/data/validation/dsv4-phase1-a5-dspark-official-$(date +%Y%m%d_%H%M%S)/native-dp4-dspark"
bash tools/dsv4/run_phase1_a5_native_smoke.sh run-dspark
```

失败时保留整个 `PHASE1_NATIVE_OUTPUT_ROOT`；不要手工改写 checkpoint 配置或屏蔽 clean-tree 审计。
