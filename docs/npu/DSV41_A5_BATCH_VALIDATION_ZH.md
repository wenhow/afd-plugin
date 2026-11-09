# A5 集中验证：V4.1 HCCL 与 Graph 通信组件

这些项目可在后续统一执行。开发期间已在 A3 验证通信组件；A5 结果用于确认
950DT/CANN 环境上的行为。下述探针不加载模型权重，不能代替完整模型、Engram、
原生 MoE、DSpark、PD 或性能验收。V4.1 服务的模型验收门禁仍保留。

完整 checkpoint 的启动及量化修复验证继续按
[原生 eager 操作说明](../../openspec/changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)
执行。下面这组组件检查可独立于完整模型启动进行。

## 更新分支

在已有 afd-plugin 仓库执行。先查看本地改动；如有需要保留的改动，先提交或备份，
再切换分支。以下步骤不清除工作区内容。

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
```

沿用容器已部署的依赖和 CANN 环境。Python 使用
`/usr/local/python3.11.10/bin/python`，无需新建 venv，也不要求名为 `atb-env` 的环境。

## 一次运行四个组件用例

在仓库根目录执行，确认 `afd_model` 是本机 checkpoint 目录。
`npu-smi info` 确认所选设备空闲后运行；示例使用设备 0、1、2、3。
两个 HCCL 服务用例和 native manager 用例需要 checkpoint 的配置文件，
不读取模型权重。static graph 用例完全不依赖 checkpoint。

```bash
afd_python=/usr/local/python3.11.10/bin/python
afd_model=/home/models/DeepSeek-V4.1-Flash
afd_repo="$PWD"
afd_batch_run="$afd_repo/.validation/a5-components-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$afd_batch_run"
export PYTHONPATH="$afd_repo:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling

git rev-parse HEAD > "$afd_batch_run/git-head.txt"
npu-smi info > "$afd_batch_run/npu-before.txt"

run_component() {
  local case_name="$1"
  shift
  if "$afd_python" -B "$@" > "$afd_batch_run/$case_name.log" 2>&1; then
    printf '%s\t0\n' "$case_name" >> "$afd_batch_run/exit-codes.tsv"
  else
    local case_status=$?
    printf '%s\t%s\n' "$case_name" "$case_status" >> "$afd_batch_run/exit-codes.tsv"
  fi
}

run_component c4-1a1f tests/npu/hccl_staged_probe.py \
  --model "$afd_model" --topology 1a1f --devices 0,1 \
  --local-counts-service --output-dir "$afd_batch_run/c4-1a1f"

run_component c4-2a2f-tp2 tests/npu/hccl_staged_probe.py \
  --model "$afd_model" --topology 2a2f-tp2 --devices 0,1,2,3 \
  --local-counts-service --output-dir "$afd_batch_run/c4-2a2f-tp2"

run_component c5-static-graph tests/npu/hccl_graph_probe.py \
  --devices 0,1 --output-dir "$afd_batch_run/c5-static-graph"

run_component c5-native-manager tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --output-dir "$afd_batch_run/c5-native-manager"

npu-smi info > "$afd_batch_run/npu-after.txt"
cat "$afd_batch_run/exit-codes.tsv"
tar -czf "$afd_batch_run.tar.gz" -C "$(dirname "$afd_batch_run")" \
  "$(basename "$afd_batch_run")"
printf '结果包：%s.tar.gz\n' "$afd_batch_run"
```

每个探针均有自己的进程监督和超时，只回收其创建的子进程。每项结果独立记录，
单项失败仍保留日志并继续收集其他项。请保留整份结果包，无需复制模型或源码。

## 看哪些结果

| 用例 | 验证内容 | 成功结果 |
| --- | --- | --- |
| `c4-1a1f` | 真实本地计数提交、连续 U1/U2、空 parent、F 服务派发和 STOP | `result.json` 的 `success=true`，各角色退出码为 0 |
| `c4-2a2f-tp2` | 两 A 报告一致、两 F 服务、native TP2 collective、STOP 后 TP 仍可用 | `result.json` 的 `success=true`，各角色退出码为 0 |
| `c5-static-graph` | 固定 BF16/FP32/IDs/F2A buffer、U1/U2/空 stage、capture 后三次更新并 replay | `summary.json` 为 passed，两个 rank JSON 的所有 case 通过 |
| `c5-native-manager` | 真实 base manager/runner、原生 InputBatch 切片、早期 GraphControl；stage bank 跨层复用、图内 padding IDs/F output mask、地址与字节预算核对；四 geometry 共 12 次 replay、6 次全局 eager fallback | `summary.json` 的 `status=passed`、`exit_codes=[0,0]`，两个 `rank-*.json` 的 `status=passed` |

回传不便时，最小结果为 `git-head.txt`、`exit-codes.tsv`，以及四项各自的结果：

- `c4-1a1f/result.json`
- `c4-2a2f-tp2/result.json`
- `c5-static-graph/summary.json` 和该目录的 `rank-*.json`
- `c5-native-manager/summary.json` 和该目录的 `rank-*.json`

同时保留体积很小的 `npu-before.txt` / `npu-after.txt`；失败项再补该项目的顶层
日志和角色日志。A3 证据及实际范围见
[C4 记录](../../openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/README.md)
、[C5 静态组件审查](../../openspec/changes/afd-hccl-v2-graph/evidence/graph-component-independent-review-20261109.md)
和 [native manager/controller 006 原始记录](../../openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/native-manager-006/results/probe/summary.json)。
追加的 [stage bank 007 记录及空闲采样窗口限制](../../openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/native-manager-bank-007/interpretation.json)
分别保留计算/内部监督通过、首次七板空闲与后续八板空闲的原始事实。

第 4 项是受控算术组件验证：早期 GraphControl 与真实 InputBatch 几何路径已执行，
不包含真实 ModelState、`ModelAclGraphManager`、Engram、原生 MoE 或完整模型。
本组也不覆盖 Engram captured producer、shared-KV 更新、GE/Dynamo lowering、
多流重叠或完整模型数值对照。当前不提供绕过模型门禁的启动参数。
