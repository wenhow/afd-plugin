# A5 并行验证：已完成场景与当前补充项

2026-11-12 已完成：**原生 eager 基线一个短请求、原生 DSpark N=1 两个连续短请求**。
三个请求均返回 32 个 token，输出 token IDs 一致；两个 DSpark 请求各有 19 个 draft、
13 个接受，覆盖实际接受和拒绝。本轮无需重跑这三个请求。

已通过的是 950DT 八卡（每卡 98304 MiB）、DP8/TP1/EP8、CPU Engram 和完整
checkpoint 的原生路径。两次 DSpark 请求落在不同 DP worker，退出清理记录仍需补齐；
AFD、PD、Graph/U2、同 worker 跨请求状态和性能仍待各自验证。

当前 A5 使用 Ascend `4ec6dc7722ec4f7f8005a86570238ac4377c5d43`，开发锁为
`7e37e29a7803d6897a7113088e2b771f83205bc0`。2026-11-12 回传的补充采集包已核验：量化
`w4a8_mxfp4.py`、DSpark `speculator.py` 及新增依赖 `ascend_forward_context.py`、
`ops/fused_moe/dataclass/fused_experts.py` 四份安装源码均与该实际版本一致。
源码采集已完成，无需重复。开发锁保持不变，退出收尾仍单独待确认。

## 下一场景：一个完整 AFD eager/U1 请求

已提供[受控 AFD 验证入口](DSV41_AFD_EAGER_QUALIFICATION_ZH.md#最小-a5-操作只新增一个-afd-请求)：
先只预检，再显式 `--run` 自动启动 A4/F4、发送一个固定 32-token 请求、先停 A、
等待 F 的 STOP 和清理，最后生成一个诊断归档。DSpark 暂时关闭；这是完整 AFD
候选的第一道硬件门，不是正式支持声明。保留当前成功环境，不安装依赖或改上游。

## 当前补充二：退出记录

如果 DSpark 服务仍在运行，先在原启动终端 Ctrl+C，等待启动器返回；若已经停止，
无需重启服务。将变量改为**既有 DSpark 运行目录**后采集当前状态：

```bash
afd_native_run=/absolute/path/of/existing/dspark/run
date -u > "$afd_native_run/post-exit-time.txt"
npu-smi info > "$afd_native_run/npu-post-exit.log" 2>&1
ps -eo pid,ppid,lstart,stat,comm > "$afd_native_run/processes-post-exit.log"
```

回传这些记录及已有的 `launcher-exit-code.txt`、`server-exit-code.txt`（若生成）。
同时简述基线服务最终释放 worker 的大致时间，以及期间是否人工停止进程或重启容器；
没有记录的部分标为未知。后补快照代表采集时刻，不能倒推之前退出时已清理完成。
本轮不新增请求，不自动停止其他进程。

## 以下为已完成的两场景操作记录

保留下面步骤供后续复现；当前无需从第 2 节重新执行。使用已配置 CANN 的容器终端；
Python 固定为 `/usr/local/python3.11.10/bin/python`。

不需要 venv 或 `atb-env`，不安装依赖，沿用容器现有 FastAPI。先确认八卡空闲、
主机 RAM 和 `/dev/shm` 足够（既有布局使用 512 GiB shm）。任何阶段失败即保留目录，
停止后续场景；不改 checkpoint、并行布局或门禁来绕过失败。

## 1. 更新分支

在 afd-plugin 源码根目录执行。若已跟踪文件有修改、远端不符或 Git 拒绝更新，保留现场；
不要执行 reset/clean。未跟踪的结果目录可以保留，Git 遇到路径覆盖冲突会停止。
以下更新块遇错即停，不会自动继续切换或启动服务。

```bash
(
  set -eu
  git remote -v
  afd_remote_url=$(git remote get-url origin)
  case "$afd_remote_url" in
    git@github.com:wenhow/afd-plugin.git|https://github.com/wenhow/afd-plugin.git) ;;
    *) printf '%s\n' 'origin 与预期仓库不符；保留现场并停止。' >&2; exit 1 ;;
  esac
  test -z "$(git status --porcelain --untracked-files=no)"
  git fetch origin feat/dsv41-flash-hccl-v030
  if git show-ref --verify --quiet refs/heads/feat/dsv41-flash-hccl-v030; then
    git switch feat/dsv41-flash-hccl-v030
  else
    git switch --track origin/feat/dsv41-flash-hccl-v030
  fi
  git merge --ff-only origin/feat/dsv41-flash-hccl-v030
  git rev-parse HEAD
)
```

`origin` 应指向 `wenhow/afd-plugin`。下面两次启动均使用同一 Python、模型路径和
已验证的 vendor 根目录；路径不同时填写本机实际值。启动器自动进行库符号和
checkpoint config/index 检查，全部日志写入新建目录，控制台只打印 `results: ...`。

## 2. 场景 A：DSpark 关闭，一个 32-token 请求

终端一启动，保持前台运行：

```bash
AFD_PYTHON=/usr/local/python3.11.10/bin/python \
AFD_MODEL=/home/models/DeepSeek-V4.1-Flash \
AFD_TRANSFORMER_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
bash tools/diagnostics/run_dsv41_a5_native_eager.sh
```

终端二进入同一源码根目录，将 `afd_native_run` 改为刚打印的真实绝对目录。
待服务就绪后执行；health 失败时先确认仍在加载，不能跳过检查继续请求。

```bash
afd_native_run=/absolute/path/printed/above
(
  set -eu
  curl --silent --show-error --fail-with-body --max-time 10 \
    http://127.0.0.1:18030/health > "$afd_native_run/health.txt"
  cat > "$afd_native_run/request.json" <<'JSON'
{"model":"dsv41-native","prompt":"Explain how to keep a simple daily work log.","temperature":0,"seed":0,"max_tokens":32,"ignore_eos":true,"n":1,"return_token_ids":true,"stream":false}
JSON
  curl --silent --show-error --fail-with-body --max-time 300 \
    -H 'Content-Type: application/json' \
    --data-binary "@$afd_native_run/request.json" \
    http://127.0.0.1:18030/v1/completions > "$afd_native_run/response.json"
  /usr/local/python3.11.10/bin/python - "$afd_native_run/response.json" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
choice = response["choices"][0]
assert choice["finish_reason"] == "length", choice
assert len(choice["token_ids"]) == 32, choice
print("baseline: 32 output token IDs saved")
PY
) > "$afd_native_run/request.log" 2>&1
afd_request_status=$?
printf '%s\n' "$afd_request_status" > "$afd_native_run/request-exit-code.txt"
printf 'request exit=%s\n' "$afd_request_status"
```

要求请求退出码为 0，响应含 32 个 token IDs。随后在终端一 Ctrl+C，等待启动器返回，
再在终端二执行一次退出复查：

```bash
npu-smi info > "$afd_native_run/npu-post-exit.log" 2>&1
```

确认八卡已无本次 worker 后才进行 B。人工 Ctrl+C 的退出码 130 单独记录；
若 worker 残留、无法退出或启动失败，停止本轮并回传现场，不连续启动第二个服务。

## 3. 场景 B：DSpark N=1，连续两个相同短请求

终端一以全新进程启动；自动创建不同结果目录。完整 draft 增加权重、KV 和 workspace，
不能从 A 的显存占用推断 B 一定可装入。

```bash
AFD_PYTHON=/usr/local/python3.11.10/bin/python \
AFD_MODEL=/home/models/DeepSeek-V4.1-Flash \
AFD_TRANSFORMER_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
bash tools/diagnostics/run_dsv41_a5_native_eager.sh --dspark 1
```

终端二把变量换成 B 打印的新目录，待服务就绪后运行。工具先检查 health，使用与 A
相同的 prompt、seed 和采样参数顺序发出两个 32-token 请求，首个失败即停止。

```bash
afd_native_run=/absolute/path/printed/by/dspark/above
/usr/local/python3.11.10/bin/python -B tools/diagnostics/collect_dsv41_dspark_requests.py \
  --output-root "$afd_native_run" --spec-tokens 1 --max-tokens 32 32 \
  > "$afd_native_run/request-collector.log" 2>&1
afd_request_status=$?
printf '%s\n' "$afd_request_status" > "$afd_native_run/request-exit-code.txt"
cat "$afd_native_run/request-collector.log"
printf 'request exit=%s\n' "$afd_request_status"
```

要求退出码为 0，`dspark-requests-*/summary.json` 中两次请求均有效，并有真实 proposal
和完整逐步 accepted/drafted 指标；**接受数量为 0 也可成立，缺指标不能记为通过**。
保留两次输出用于对照 A；不以 token 相同或 HTTP 200 单独判断 DSpark 正确性。
完成后按 A 同样 Ctrl+C、等待退出、保存 `npu-post-exit.log` 并检查八卡。

## 4. 最少回传文件

先提供两次启动目录的名字以及：

- 两个目录各自的 `preflight.json`、`environment.txt`、`request-exit-code.txt`、
  `launcher-exit-code.txt`、`server-exit-code.txt`（若已生成）和 `npu-post-exit.log`。
- A 的 `request.json`、`response.json`、`request.log`。
- B 的 `dspark-requests-*/summary.json`、`exit-code.txt` 和 `request-collector.log`。
- 若失败，再提供匹配时间的 `native-server.log` 最早 worker 根 traceback；若启动前失败，
  提供 `launcher.log` 及对应的 `preflight.log` 或 `opapi-preflight.json` / `.log`。

原始 `native-server.log`、B 的各请求响应及 metrics 全文全部保留在环境，不必首轮全量上传。
本轮结果用于原生路径定位，不构成 AFD/PD、Engram 内部历史、完整精度或性能验收。
详细边界见 [原生 DSpark 说明](DSV41_A5_NATIVE_DSPARK_ZH.md) 和
[开发验证说明](DSV41_DSPARK_VALIDATION_ZH.md)。
