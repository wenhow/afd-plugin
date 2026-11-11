# A5 原生 DSpark eager 对照

完整 draft 会增加权重、KV 和 workspace 显存；既有 DSpark-off 的显存数据不能保证开启后仍能装入。先在空闲的 950DT 八卡环境手工执行 **N=1 单轮 eager**，保存失败现场或完整结果，再决定是否进行 N=2/3。本文命令尚未在 A5 执行，不代表 DSpark、AFD 或完整模型验收通过。

这一路使用原生 vLLM/Ascend，保持现有 DP8/TP1/EP8、PP/PCP/DCP=1、Engram CPU offload/DP shared memory、小请求和同一完整 checkpoint。不开启 AFD 插件、Graph、DBO、PD 或 TP8 量化兼容补丁，不修改上游或安装 Python 包。AFD 完整模型门禁保持原状。

## 更新并启动一轮

在现有 CANN shell、afd-plugin 源码根目录使用 `/usr/local/python3.11.10/bin/python`。保留本地修改，更新遇到分叉时不要 reset/clean。

```bash
git status --short
git fetch origin feat/dsv41-flash-hccl-v030
if git show-ref --verify --quiet refs/heads/feat/dsv41-flash-hccl-v030; then
  git switch feat/dsv41-flash-hccl-v030
else
  git switch --track origin/feat/dsv41-flash-hccl-v030
fi
git merge --ff-only origin/feat/dsv41-flash-hccl-v030
git rev-parse HEAD
bash tools/diagnostics/run_dsv41_a5_native_eager.sh --help
AFD_PYTHON=/usr/local/python3.11.10/bin/python \
AFD_MODEL=/home/models/DeepSeek-V4.1-Flash \
AFD_TRANSFORMER_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
bash tools/diagnostics/run_dsv41_a5_native_eager.sh --dspark 1
```

`AFD_TRANSFORMER_VENDOR` 指向当前已验证的 vendor 根目录；若安装位置不同，应使用本机实际目录。脚本保留既有 CompressorV2 库检查和 checkpoint config/index 校验。不创建 venv，不替换 FastAPI 0.123.10 或其他已部署依赖。

控制台打印新的 `results/dsv41-native-dspark-n1-dp8-ep8.XXXXXX`，服务前台运行，日志全部写入该目录。启动失败时保留目录并停止本轮，不连续尝试其他 N。确认日志中的实际 draft 选择、并行组、量化与内存分配；文件存在性和源文件 hash 只提供身份信息。

脚本仅在显式 `--dspark 1|2|3` 时追加：

```json
{"method":"dspark","num_speculative_tokens":1,"draft_tensor_parallel_size":1,"enforce_eager":true}
```

并启用 `--per-request-spec-decode-metrics detailed`。若安装版本不支持这些原生字段或 CLI，保留启动错误；不要替换成普通 MTP 或删除校验继续运行。无参数仍是原来的 DSpark-off 基线；`--probe-only` 保持独立，不能与 `--dspark` 合用。

N 表示本轮 speculative token 数，target 验证宽度为 N+1，draft 根据 checkpoint 的 `sample_from_anchor` 使用 N 或 N+1 query；它不是 draft 层数或卡数。锁定 vLLM 的 DSpark loader 从 target 的 ParallelConfig 派生 draft，保留 DP/EP，修改 draft TP/PP 并关闭 draft EPLB。这里 target TP1 与 draft TP1，实际仍需八卡原生 DP8/EP8，不是单卡 draft 或两卡 A/F。

## 请求、指标和退出

在另一个终端设置刚才打印的真实结果目录。先检查 `/health`，再执行一个固定 seed 的单序列请求；`ignore_eos` 仅用于形成固定长度 decode，不强制接受 draft。

```bash
afd_native_run=/absolute/path/printed/above
(
  set -eu
  curl --silent --show-error --fail --max-time 10 \
    http://127.0.0.1:18030/health > "$afd_native_run/health.txt"
  curl --silent --show-error --fail --max-time 10 \
    http://127.0.0.1:18030/metrics > "$afd_native_run/metrics-before.txt"
  npu-smi info > "$afd_native_run/npu-ready.log" 2>&1
  cat > "$afd_native_run/request.json" <<'JSON'
{"model":"dsv41-native","prompt":"Explain how to keep a simple daily work log.","temperature":0,"seed":0,"max_tokens":64,"ignore_eos":true,"n":1,"return_token_ids":true,"stream":false}
JSON
  curl --silent --show-error --fail-with-body --max-time 300 \
    -H 'Content-Type: application/json' \
    --data-binary "@$afd_native_run/request.json" \
    http://127.0.0.1:18030/v1/completions > "$afd_native_run/response.json"
  curl --silent --show-error --fail --max-time 10 \
    http://127.0.0.1:18030/metrics > "$afd_native_run/metrics-after.txt"
  npu-smi info > "$afd_native_run/npu-after-request.log" 2>&1
) > "$afd_native_run/request.log" 2>&1
afd_request_status=$?
printf '%s\n' "$afd_request_status" > "$afd_native_run/request-exit-code.txt"
```

如需实际观测 HBM 峰值，可在第三个终端于请求前启动这段有界监控；其超时退出码 124 表示正常到达采集时限。它不终止模型服务。请求若超过窗口，不能把窗口内最大值称为整轮峰值。

```bash
afd_native_run=/absolute/path/printed/above
timeout 120s bash -c 'while true; do date -u; npu-smi info; sleep 1; done' \
  > "$afd_native_run/npu-sampled.log" 2>&1
```

保存实际响应中的 `metrics.speculative_decoding`（若安装版本已实现该响应字段）、输出 token IDs、finish reason 和请求退出码。锁定版本的有序数组字段为 `per_step_accepted` / `per_step_drafted`，另需查看接受长度直方图、draft 总数及接受总数。保留 `/metrics` 全文及 engine/rank 标签，比较以下计数器的请求前后增量：

- `vllm:spec_decode_num_drafts_total`
- `vllm:spec_decode_num_draft_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_per_pos_total`

draft 接受率为 accepted/drafted；平均接受长度按原生口径含 bonus token，为 `1 + accepted/drafts`。缺指标不能记为零，也不能据 HTTP 200 推断发生了有效 proposal。原生 DP8 的 dummy 协调和每个 engine 的计数须按实际标签保存。

成功后再用同一服务执行不同长度和重复 prompt 的独立请求，分别保存 JSON、响应与指标快照；不要覆盖第一次请求。随后在服务终端 Ctrl+C，等待退出码与 `npu-after.log`，再检查八卡是否空闲、原 worker 是否仍存活。未退出或残留必须如实记录，不能只凭请求成功认定已回收。

应回传新结果目录中的 `environment.txt`、`preflight.json`、`opapi-preflight.json`、`native-server.log`、`launcher.log`、请求/响应/指标、显存采样和退出码文件。`preflight.json` 新增实际 DSpark、rejection、Engram 与指标源码 hash；实际 A5 上游版本与项目锁不同须保留差异。Engram 使用主机内存，同时记录 RAM 与 `/dev/shm`，不要只看 HBM。

完成 N=1 的数据、退出和资源复核后，N=2/3 可各用同一入口启动全新进程、全新结果目录。DSpark-off 对照也使用全新进程及相同 prompt、seed、长度和并发布局。token 对照与计数可以暴露行为差异，但不能直接证明 Engram/KV 内部历史正确，更不能替代 AFD target-only、多流或 Graph/PD 组合验收。整体开发边界见 [DSpark 验证说明](DSV41_DSPARK_VALIDATION_ZH.md)。
