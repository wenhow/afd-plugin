# A5 集中验证：V4.1 HCCL 与 Graph 通信组件

这些项目可在后续统一执行。开发期间已在 A3 验证通信组件；A5 结果用于确认
950DT/CANN 环境上的行为。下述探针不加载模型权重，不能代替完整模型、Engram、
原生 MoE、DSpark、PD 或性能验收。V4.1 服务的模型验收门禁仍保留。

完整 checkpoint 的启动及量化修复验证继续按
[原生 eager 操作说明](../../openspec/changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)
执行。下面这组组件检查可独立于完整模型启动进行。

DSpark 优先开发的执行边界、CPU 回归与后续整模型验收见
[DSpark 验证说明](DSV41_DSPARK_VALIDATION_ZH.md)。

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

## 一次运行十七个组件用例

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

run_component c6-pipeline-hybrid1 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --compute-pipeline --pipeline-hybrid 1 \
  --output-dir "$afd_batch_run/c6-pipeline-hybrid1"

run_component c6-pipeline-hybrid0 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --compute-pipeline --pipeline-hybrid 0 \
  --output-dir "$afd_batch_run/c6-pipeline-hybrid0"

run_component c6-eager-isend tests/npu/hccl_eager_backend_probe.py \
  --model "$afd_model" --devices 0,1 --operation isend \
  --output-dir "$afd_batch_run/c6-eager-isend"

run_component c6-eager-irecv tests/npu/hccl_eager_backend_probe.py \
  --model "$afd_model" --devices 0,1 --operation irecv \
  --output-dir "$afd_batch_run/c6-eager-irecv"

run_component c6-axes11111 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --graph-compute 1 --graph-hybrid 1 --attention-three 1 --ffn-recv 1 --ffn-cross 1 \
  --output-dir "$afd_batch_run/c6-axes11111"

run_component c6-axes00000 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --graph-compute 0 --graph-hybrid 0 --attention-three 0 --ffn-recv 0 --ffn-cross 0 \
  --output-dir "$afd_batch_run/c6-axes00000"

run_component c6-axes10000 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --graph-compute 1 --graph-hybrid 0 --attention-three 0 --ffn-recv 0 --ffn-cross 0 \
  --output-dir "$afd_batch_run/c6-axes10000"

run_component c6-axes10010 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank \
  --graph-compute 1 --graph-hybrid 0 --attention-three 0 --ffn-recv 1 --ffn-cross 0 \
  --output-dir "$afd_batch_run/c6-axes10010"

run_component c6-eager-owner-tp1 tests/npu/hccl_eager_pipeline_probe.py \
  --model "$afd_model" --tp-size 1 --devices 0,1 --stage-diagnostics \
  --output-dir "$afd_batch_run/c6-eager-owner-tp1"

run_component c6-eager-owner-tp2 tests/npu/hccl_eager_pipeline_probe.py \
  --model "$afd_model" --tp-size 2 --devices 0,1,2,3 --stage-diagnostics \
  --output-dir "$afd_batch_run/c6-eager-owner-tp2"

run_component c6-graph-bulk-v1 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,1 --stage-bank --exchange-order bulk_v1 \
  --graph-compute 1 --graph-hybrid 1 --attention-three 1 --ffn-recv 1 --ffn-cross 1 \
  --output-dir "$afd_batch_run/c6-graph-bulk-v1"

run_component c6-graph-cross-layer-v1 tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,2 --stage-bank \
  --exchange-order cross_layer_v1 --layer-count 3 \
  --graph-compute 1 --graph-hybrid 1 --attention-three 1 --ffn-recv 1 --ffn-cross 1 \
  --output-dir "$afd_batch_run/c6-graph-cross-layer-v1"

run_component c7-target-query3-cross tests/npu/hccl_graph_manager_probe.py \
  --model "$afd_model" --devices 0,2 --stage-bank --target-query-len 3 \
  --exchange-order cross_layer_v1 --layer-count 3 \
  --graph-compute 1 --graph-hybrid 1 --attention-three 1 --ffn-recv 1 --ffn-cross 1 \
  --output-dir "$afd_batch_run/c7-target-query3-cross"

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
| `c6-pipeline-hybrid1` | 在上述固定 bank 上捕获侧流计算、父流 HCCL、逐 stage 接收依赖和最终 join；同一 key 三次 replay 的有效 token 数为完整桶、1、桶大小减 1，并核对跨层递推及末层结果 | `summary.json` 的 `status=passed`、`exit_codes=[0,0]`；两个角色的 `compute_pipeline=true`、`pipeline_hybrid=1` |
| `c6-pipeline-hybrid0` | 独立 session 验证完整父流 frontier 依赖，其他输入及数值检查与 hybrid1 相同 | 同上，角色的 `pipeline_hybrid=0` |
| `c6-eager-isend` | 同一已预热 HCCL 组上，三个 dtype 的异步发送 host 返回、Work 依赖、发送 buffer 复用及对端原始数据 | `result.json` 的 `capability_gate_passed=true`、`all_payloads_drained_and_session_closed=true`，`exit_codes` 的两个角色均为 0 |
| `c6-eager-irecv` | 独立 session 检查接收投递、等待及消费者在匹配前保持依赖，匹配后数值精确 | 同上；它们是后端能力检查，尚不代表 eager/U2 模型流水验收 |
| `c6-axes11111` | 五个 Graph 配置轴全部开启，实际 U2 按 selection 分配 pipeline；U1 保持 pipeline None | `summary.json` 的 `status=passed`、`exit_codes=[0,0]`；角色 JSON 中 `graph_axes` 为对应五值、`graph_selections` 的 U1 `pipeline_allocated=false`，U2 为 true |
| `c6-axes00000` | 五轴全关；相同递推、有效 token 变化及末层校验仍走 Graph/U2，并非改为 eager | 同上；两个角色 `pipeline_storage=[]`，八个 warmup/capture `graph_selections` 都为 `pipeline_allocated=false`，`transactions` 仍含 4 个 capture、12 个 replay |
| `c6-axes10000` | 仅 compute 开启；hybrid/Attention-three/F-recv/F-cross 关闭，F 每层完成 join | 同上；U2 的 `pipeline_storage` 中四个 policy 字段全为 false，仍有侧流 compute |
| `c6-axes10010` | 在上一项基础上开启 F-recv，保留关闭跨层流水时的每层 join | 同上；U2 policy 的 `ffn_recv_stream=true`、`ffn_cross_layer=false` |
| `c6-eager-owner-tp1` | 两 stage group 显式匹配预热，真实 owner/ticket/loan；6 个 target U2 parent、双层递推、在途上限与同 stage 复用，原始 FP32、raw IDs、完整 padding 输出及空 stage/全空 parent | `result.json` 的 `success=true`，`exit_codes` 两角色均为 0，`killed_owned_pids=[]`；每个角色 JSON 的 `success=true`、6 个 `parents`、16 个 `checked_steps`、`normal_stop=true`，`native_tp_before` 和 `native_tp_after_plugin_close` 均为 `[1.0]` |
| `c6-eager-owner-tp2` | 独立四卡 session 验证上述配对 TP2 owner、每角色 native TP 域及 STOP 后保留 | 同上；`exit_codes` 四角色均为 0，四个角色 JSON 的 native TP 前后均为 `[2.0]` |
| `c6-graph-bulk-v1` | 双角色显式协商 bulk_v1；U2 每层 A 全发送后全接收，F 全接收并排入计算后全发送，HCCL 保持 capture parent；U1 和 eager fallback 保持串行 | `summary.json` 的 `status=passed`、`exit_codes=[0,0]`、`exchange_order=bulk_v1`；两个 `rank-*.json` 的 `status=passed`、26 个 `transactions`，U2 `pipeline_storage[*].exchange_order=bulk_v1`，native TP 前后均为 `[1.0]` |
| `c6-graph-cross-layer-v1` | 三层 TP1，双角色显式协商 cross_layer_v1，执行生产共享 12 步跨层 wire schedule；F 在 compute_done 前复制到 stage-owned output，A 消费前层 reply 后再复用；保留原 FP32/IDs/地址/递推 oracle | `summary.json` 的 `status=passed`、`exit_codes=[0,0]`、`exchange_order=cross_layer_v1`、`layer_count=3`；各 `rank-*.json` 为 passed、26 个 transactions，4 个 U2 pipeline_storage 各含 12 条 static_wire_actions、6 个 positive_steps；native TP 前后均为 `[1.0]` |
| `c7-target-query3-cross` | 固定 q3 的三层组件；T9/R3、T12/R4 的 U1/U2；额外 R2/T6 同 key replay 与 R1/T3 clamp miss | summary 和两个 rank 均 passed、退出码 `[0,0]`；各 rank `target_query_len=3`、27 个 transactions、唯一 short-batch replay 的 valid=`[4,2]`、padded=`[4,5]`；native TP 前后 `[1.0]` |

五轴顺序是 compute、hybrid、Attention-three、F-recv、F-cross。五个 CLI 必须一起显式
传入，不能与第 5～6 项的 legacy 参数混用；`--ffn-cross 1` 要求 `--ffn-recv 1`。
每项使用固定 session 设置和独立 cache，不能在同一 captured graph 上修改开关。
`graph_selections[*].selection` 记录 configured/effective 以及 logical/physical stream；
Graph 通信始终受父流约束，不应将逻辑 send/recv 配置解释为物理独立通信流。

回传不便时，最小结果为 `git-head.txt`、`exit-codes.tsv`，以及十七项各自的结果：

- `c4-1a1f/result.json`
- `c4-2a2f-tp2/result.json`
- `c5-static-graph/summary.json` 和该目录的 `rank-*.json`
- `c5-native-manager/summary.json` 和该目录的 `rank-*.json`
- `c6-pipeline-hybrid1/summary.json` 和该目录的 `rank-*.json`
- `c6-pipeline-hybrid0/summary.json` 和该目录的 `rank-*.json`
- `c6-eager-isend/result.json`
- `c6-eager-irecv/result.json`
- `c6-axes11111/summary.json` 和该目录的 `rank-*.json`
- `c6-axes00000/summary.json` 和该目录的 `rank-*.json`
- `c6-axes10000/summary.json` 和该目录的 `rank-*.json`
- `c6-axes10010/summary.json` 和该目录的 `rank-*.json`
- `c6-eager-owner-tp1/result.json`，以及 `attention-0.json`、`ffn-0.json`
- `c6-eager-owner-tp2/result.json`，以及 `attention-0.json`、`attention-1.json`、`ffn-0.json`、`ffn-1.json`
- `c6-graph-bulk-v1/summary.json`，以及 `rank-0.json`、`rank-1.json`
- `c6-graph-cross-layer-v1/summary.json`，以及 `rank-0.json`、`rank-1.json`
- `c7-target-query3-cross/summary.json`，以及 `rank-0.json`、`rank-1.json`

同时保留体积很小的 `npu-before.txt` / `npu-after.txt`；失败项再补该项目的顶层
日志和角色日志。A3 证据及实际范围见
[C4 记录](../../openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/README.md)
、[C5 静态组件审查](../../openspec/changes/afd-hccl-v2-graph/evidence/graph-component-independent-review-20261109.md)
和 [native manager/controller 006 原始记录](../../openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/native-manager-006/results/probe/summary.json)。
追加的 [stage bank 007 记录及空闲采样窗口限制](../../openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/native-manager-bank-007/interpretation.json)
分别保留计算/内部监督通过、首次七板空闲与后续八板空闲的原始事实。
新增 [eager 后端两方向 gate 记录](../../openspec/changes/afd-hccl-v2-multistream/evidence/a3-eager-backend-20261110/README.md)
与 [四个五轴组件会话](../../openspec/changes/afd-hccl-v2-multistream/evidence/a3-graph-axes-20261110/README.md)
分别记录实际 Work/event 与固定 Graph 调度轴的验证范围。

第 4～6 项是受控算术组件验证：早期 GraphControl 与真实 InputBatch 几何路径已执行，
不包含真实 ModelState、`ModelAclGraphManager`、Engram、原生 MoE 或完整模型。
本组也不覆盖 Engram captured producer、shared-KV 更新、GE/Dynamo lowering、
有效多流重叠、性能收益或完整模型数值对照。第 5～6 项只增加实际 stream/event
捕获与 replay 正确性检查；它们不验证原生 scheduler 缩减请求后仍命中同一几何。
当前不提供绕过模型门禁的启动参数。

第 9～12 项同样只验证受控组件，沿用真实 base manager、InputBatch 几何、bank、
GraphControl 和实际 NPU stream/event/HCCL。每项均有 4 次 warmup、4 次 capture、
12 次 replay 与 6 次刻意触发的 eager fallback；关闭所有 overlap 时，正常四个 key
仍应 capture/replay。它们不覆盖所有开关组合、真实 ModelState/Engram/MoE 或性能。

第 13～14 项是实际 eager ticket owner 组件，和第 7～8 项的原始后端 gate 分开。
它们全部通过公开的预热、计数提交、accepted plan、post、loan、finish 和 STOP API，
每层先投递两个 positive stage，再消费本层全部输出并进入下一层。因此它们验证
有界在途通信、BF16/独立 FP32/int32 IDs、借用依赖及同 stage buffer 复用，
不代表原生 V4.1 延迟 HC/pre_mix/Engram、原生 MoE、完整模型或重叠性能验收。
角色文件名是 `attention-*.json` / `ffn-*.json`，不是 Graph 用例的 `rank-*.json`。
这两项已带 `--stage-diagnostics`，不增加用例数量。每个角色的
`diagnostic_validation.configuration_count=6`，其他布尔校验字段均应为 `true`；
`diagnostic_records` 保存每个 accepted parent 的配置及 post、loan、reuse、finish
主机边界，包含精确的 transaction、world/role rank、实际 compute/send/recv 及 alias。
所有记录的 `actual.ffn_compute_sync=false`；全空 parent 也应有 configuration 与
finish 记录。此参数只增加标量记录和校验，不增加 device wait、event、query 或
synchronize。请务必回传上述各角色 JSON；只有顶层 `result.json` 不足以核对日志。
这些记录不代表原生模型消费者或有效重叠、性能验收。
结果和 A3 范围见 [实际 owner 记录](../../openspec/changes/afd-hccl-v2-multistream/evidence/a3-eager-pipeline-20261110/README.md)。

第 15 项是独立 Graph bulk_v1 通信组件，不加载原生模型。它保留真实 base
manager、固定 stage bank、独立 FP32/IDs/输出 oracle 和 native TP 检查，覆盖
4 warmup、4 capture、12 replay、6 eager fallback。角色 JSON 的
`native_tp_before` 与 `native_tp_after_close` 均应为 `[1.0]`；U2 的四个
`pipeline_storage` 资源记录必须为 `exchange_order=bulk_v1`。前面的既有 Graph
用例仍保持默认 `stage_serial`，不能在同一 session/cache 中切换 order。
该组件只验证同层输入前沿与实际 capture/replay 正确性，不代表物理 send/recv
侧流、完整跨层流水、原生 HC/Engram/MoE、完整模型或性能验收。A3 的单次 TP1
会话及范围见 [bulk 记录](../../openspec/changes/afd-hccl-v2-multistream/evidence/a3-graph-bulk-20261110/README.md)。

第 16 项是独立 session 的三层 `cross_layer_v1` 组件。执行前先用 `npu-smi info`
确认八卡全部空闲；本项只使用 0、2 两卡，TP1，各角色分别保留 native TP 域。
记录本地 Git SHA、来源锁 `.agents/dsv41/sources.lock.json`、实际 vLLM /
vLLM-Ascend / torch / torch_npu 版本，并审核锁定源码与容器安装源码是否一致；
有差异先保留记录并核对，不能从包版本或 A3 结果推定本机源码相同。
它不修改依赖，不加载权重，也没有绕过模型门禁的启动参数。

本项仍有 4 warmup、4 capture、12 replay、6 eager fallback。U1 保持 serial；
U2 两个正 stage 的共享静态顺序为
`I00 I01 O00 I10 O01 I11 O10 I20 O11 I21 O20 O21`，两角色均从生产 schedule
读取，`static_wire_actions` 是静态计划而非设备时间线。固定 oracle 检查独立原始
FP32、BF16、raw IDs、固定 bank 地址、完整桶 / 1 / T-1 有效 token 及末层递推，
不通过降低容差或修改期望值来适配此顺序。所有 HCCL 仍在 capture parent，
compute 在实际侧流；不能据此声称物理通信侧流、有效重叠或性能收益。

最小回传为本项的 `summary.json`、`rank-0.json`、`rank-1.json`，以及共同的
Git SHA、版本/来源锁核对记录和八卡运行前后 `npu-smi`。A3 的单次三层 TP1
通过结果及逐 rank 断言见
[cross-layer 记录](../../openspec/changes/afd-hccl-v2-multistream/evidence/a3-graph-cross-layer-20261110/README.md)。
该结果不包含原生 HC/pre_mix、Engram、Attention、MoE、完整 checkpoint、TP2
设备或完整 C6 验收。

第 17 项以固定目标查询宽度 q=3 验证同一套三层组件，不创建草稿模型。
原生 base manager 捕获 T9/R3、T12/R4 的 U1/U2 四种几何；每角色应为
4 warmup、4 capture、13 replay、6 eager fallback，共 27 个事务。原来的
full/1/T-1 有效行覆盖仍保留，额外一轮实际 R2/T6 应命中 T9/U2，stage 的
valid=(4,2)、padded=(4,5)。实际 R1/T3 因原生 request clamp 改变布局，
必须共同选择 eager；F generation miss 也必须共同回退。

最小回传同第 16 项，并确认两个角色的 `target_query_len=3`、
`expected_transactions=27`、27 条 `transactions`。Attention 的 rank-1 应有唯一
`short_batch_same_key=true` 事务；FFN 的 rank-0 应有相同 ID/key 及 valid/padded
计数的 replay。该项证明的是固定宽度几何与受控算术，
不包含真实 DSpark proposal、接受/拒绝、草稿 KV/history 或完整模型。
既有十六项仍默认 q=1，不要在同一 session 内修改查询宽度。

A3 已完成该组件的一次 TP1 实测，逐角色结果和范围见
[q3 组件记录](../../openspec/changes/afd-hccl-full-parity/evidence/a3-target-query-graph-20261110/target-query-cross-tp1-001-README.md)。
A5 的第 17 项仍需在本机独立执行，不能沿用 A3 结果作为 A5 通过记录。
