# DSV4.1 非 Engram 功能：A5 最小集中验证

本轮先收口非 Engram 功能代码，再由 A5 运行下面这一批。**一条命令、7 个顺序场景**，
包含原生对照、正常 A/F Worker 四模式，以及 DSpark/PD 两个代表组合。
不需要重新执行旧的 17 场景列表，也不安排 profiler、性能或服务启停专项。

本批的正常模型使用原生三层 `[0,0,2]` C2 Attention/HC、64个Attention头、
32个Indexer头（维度128）、原生完整 BF16 MoE、
真实 Scheduler、KV 规划和设备算子；参数为受控 dummy，Engram 关闭。
DSpark/PD 使用真实原生 DSpark draft、设备缓存写入和 Mooncake 传输，target/FFN
仍是有明确输出预期的受控组件。因此结果用于**功能组合验证**，不代表训练检查点
精度、完整 Engram 模型或所有拓扑/长运行已经验收。

## 1. 更新并执行

使用此前能运行 DSV4.1 的 A5 容器和 Python 环境，保留其 CANN/ATB/OPP 环境。
代码锁为 vLLM `ced6857a`、vLLM-Ascend `7e37e29a`。入口会核对原生 Python
源码，以及实际 A5 类型、Mooncake 和已安装原生扩展；验证包不携带 A3 二进制，
不安装包、不替换共享运行库。

```bash
git switch feat/dsv41-flash-hccl-v030
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /mnt/workspace/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --output-dir /mnt/workspace/tmp/dsv41-a5-functional-20261117 \
  --run
```

`--model` 改为现有模型目录；这里只读原始 `config.json`，受控配置写入结果目录。
`--output-dir` 必须不存在，重跑时换一个目录。默认使用当前 `python`；也可显式传
`--python /实际虚拟环境/bin/python`。`--devices` 是当前进程可见的四个不同设备序号；
正常 A/F 用前两个，DSpark/PD 四角色用四个。执行前后检查整机八卡空闲。

`--transformer-vendor` 指向现有算子vendor根目录，要求其中存在
`op_api/lib/libcust_opapi.so`。入口在启动预检和各角色子进程前，将vendor根目录
前置到`ASCEND_CUSTOM_OPP_PATH`，将其`op_api/lib`前置到`LD_LIBRARY_PATH`；
原有路径保留，原生/A/F/DSpark各进程均继承。实际选值写入`summary.json`的
`transformer_vendor`和`runtime_environment`。若容器已正确设置这两个环境变量，
可以省略该参数。仅修改本次子进程环境，不安装或替换算子库。

原生导入/注册预检的上限默认 600 秒，可用 `--preflight-timeout 900` 单独调整，
为首次导入的 JIT 编译留出时间；该参数不改变单场景 `--timeout`（默认 480 秒）。
预检超时会明确报告 `preflight timed out`，所用上限保存在 `summary.json` 的
`preflight_timeout_seconds` 中。超时、任一角色非零退出、被强制清理、报告失败、残留进程或
设备未空闲都算失败。必要的每轮清理保留。遇到失败即停止，保留首次原始日志，
不会把已写出的成功字段覆盖进程异常。

可选的源码/参数自检去掉 `--run` 即可；该路径不导入设备运行库，**不计为 A5 验证通过**。

## 2. 七个场景及通过条件

| 场景 | 要验证的功能 | 关键判定 |
| --- | --- | --- |
| 1 原生 eager/U1 | A5 原生基线 | 真实 Worker、加载、KV、warmup；A/B 两轮后取消 A，B 结束，追加 C 再结束 |
| 2 AFD eager/U1 | 正常 A/F 接入和连续轮次 | 同一正常 Worker 生命周期，真实远程 MoE；与原生实际参数和采样结果比较 |
| 3 AFD eager/U2 多流 | 原生 U2 切分和 eager 多流接入 | 实际两阶段与 `eager-streamed` 记录；取消后剩余请求继续，C 可接入 |
| 4 AFD Graph/U1 | 正常捕获、回放及 prefill eager 路径 | 实际 capture/replay 返回记录，保留同一模型/KV owner，随后请求可继续 |
| 5 AFD Graph/U2 多流 | 两阶段 Graph 正常入口 | 两请求 decode 的真实 U2 replay、实际 Graph 多流执行记录，随后单请求及新请求继续 |
| 6 DSpark/PD eager/U1 | 拒绝后的 KV/历史与 Ready | prefix/辅助状态实际传输，拒绝后继续原生 proposal/verification，Ready 前沿和 typed DONE |
| 7 DSpark/PD Graph/U2 多流/跨层 | 代表性组合与请求生命周期 | `cross_layer_v1`、全部 Graph policy 轴显式开启；实际 capture/replay、接受、取消、追加 C、缓存复用与 DONE |

前五项使用同一模型声明与参数初始化，自动核对原生 79 份参数与 A58+F21 的
canonical 名字、shape、dtype、字节 SHA，并比较 A/B/C 的真实贪心采样序列。
不会预先写死正常模型的 token ID。DSpark 组件的受控 target 则检查事先声明的
接受/拒绝结果，不能把这些 token 当作完整模型精度证据。

多流场景检查实际执行路径和协议完成；没有设备 timeline，不能据此宣布物理重叠或性能收益。
Graph 场景同时经过 prefill 的 eager 路径；本批不额外扩成所有 bucket、miss 原因或拓扑组合。
P 端中途内存压力抢占仍按现有协议终止，不声称支持断点恢复；请求取消、正常退役和随后请求
是本批覆盖的生命周期。

## 3. 返回结果

首先看进程退出码和结果目录的 `summary.json`：

```bash
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print({k:r[k] for k in ("success","npu_acceptance","runtime_source_sha256")})' \
  /mnt/workspace/tmp/dsv41-a5-functional-20261117/summary.json
```

本批通过需要退出码 `0`、`success=true`、`npu_acceptance=component_suite_passed`，
且七场景均完成。正常模型比较位于 `native_afd_comparison`；每轮命令、角色退出码、
实际阶段、参数、采样历史、KV 描述、原始 hidden 和清理结果都保留在各场景目录。

请回传整个结果目录的压缩包，无需手工筛日志：

```bash
tar -czf dsv41-a5-functional-20261117-results.tar.gz \
  -C /mnt/workspace/tmp dsv41-a5-functional-20261117
```

如果仍在 `SparseFlashMlaMetadata` 失败，保留这次结果，不先改模型几何或替换算子库。
A3 已知是首轮与原生一致、第二轮 decode C2 metadata 失败；本批 A5 原生/AFD 同参数
对照用于区分这个差异。**目前不能确认 A3 环境就是根因。**

### 启动脚本与冻结包不匹配时

`launcher differs from the frozen bundle` 表示当前
`tests/npu/dsv41_a5_functional_validation.py` 的文件 SHA256 与同包
`bundle-lock.json` 的 `launcher_sha256` 不一致。此时归档及包内文件校验已通过，
尚未创建结果目录，也未开始原生源码预检或设备场景；与 vLLM-Ascend 的 Git 版本核对分开。
脚本内容改动、换行转换或脚本/归档混用版本都可能触发，需比较实际文件确定原因。

在 AFD 仓库根目录查看实际值和期望值：

```bash
sha256sum tests/npu/dsv41_a5_functional_validation.py
python -B -c 'import json; print(json.load(open("reports/dsv41-a5-functional/bundle-lock.json"))["launcher_sha256"])'
```

默认交付包的入口 SHA256 为
`b0c6bba5b068efdd35aa52b79eb825ff8943e9c4e5e5cb89c9f05ba2889a11e7`。
更新时需要同步入口与 `reports/dsv41-a5-functional/` 下的归档、锁和准备记录，
不要只复制新增超时参数后的脚本。
保留现场脚本，再从已经通过校验的同一冻结包恢复入口：

```bash
launcher=tests/npu/dsv41_a5_functional_validation.py
backup=$(mktemp "${launcher}.before-frozen.XXXXXX")
cp -p "$launcher" "$backup" &&
  tar -xzf reports/dsv41-a5-functional/runtime-source.tar.gz "$launcher"
```

如果执行时显式传了 `--source-bundle`，上述归档和锁文件路径也应改为那一份。
恢复后重新执行第 1 节命令即可；若此次仅停在入口校验，同一个输出目录仍未创建，可以复用。
无需在 A5 重建冻结包或修改 `launcher_sha256`；该恢复只涉及 AFD 验证入口，原生仓库不变。

### 原生运行时预检失败时

外层 `A5 native/runtime preflight failed` 是汇总错误。具体异常写入结果目录的
`native-runtime.json`；退出码、超时与强制清理记录在 `native-runtime.process.json`，
同一进程结果也保存在 `summary.json` 的 `runtime_preflight` 中。
`native-runtime.log` 只有导入/编译输出时，不能根据最后一条 warning 判断失败原因。
无需重跑场景，先读取已有记录：

```bash
python -B - <<'PY'
import json
from pathlib import Path

root = Path('results/dsv41-a5-functional-20261117')  # 改为实际输出目录
for name in ('native-runtime.process.json', 'native-runtime.json', 'summary.json'):
    path = root / name
    print('\n' + str(path))
    if not path.exists():
        print('missing')
        continue
    report = json.loads(path.read_text())
    fields = ('status', 'traceback', 'sources_match', 'source_count',
              'returncode', 'timed_out', 'forced_killed_pids',
              'surviving_owned_pids', 'elapsed_seconds',
              'preflight_timeout_seconds', 'runtime_preflight')
    print(json.dumps({k: report[k] for k in fields if k in report}, indent=2))
PY
```

原始交付入口的预检上限固定为 120 秒；11月17日修复后默认 600 秒，
可由 `--preflight-timeout` 调整，独立于单场景 `--timeout`。如果 `timed_out=true`，
表示预检被外层超时清理；首次导入的 JIT 编译可能占用该时限，需结合日志确认，
不能将这种退出直接归因为算子缺失。若 `timed_out=false` 且有 `traceback`，
按其中的原始异常定位。源码身份检查在运行库导入前完成；该检查失败时，先核对实际安装目录对应的锁定源码。

11月17日第二次回传已确认 `sources_match=true`、194份源码全部匹配，
`timed_out=true`、`returncode=-9`；日志停在 JIT 编译输出，尚未进入七个功能场景。
这是预检超时退出，不能据此认定缺失算子是根因或功能已通过。报告还在清理采样时记录了
一个 `surviving_owned_pids`；保留为失败证据，不能宣称该次清理成功，也不能仅凭旧PID
推断进程现在仍存在。旧结果目录保留，重跑使用新目录。

### 正常场景配置阶段的 NPU 初始化

如果旧结果在 `dsv41_functional_worker_probe.py` 的
`assert report["npu_initialized_after_config"] is False` 失败，查看同一报告的
`configuration`、`model_loaded` 和 `inference_executed`。11月17日回传已确认
原生预检通过，正常原生配置构造完成，但配置查询使NPU初始化，随后被验证脚本的CPU断言阻断，
当时模型尚未加载。

修复后的 `--run` 会在配置前绑定指定的进程可见设备，记录
`configuration_device_before/after`，并检查配置后仍为同一设备，再进入原生Worker。
`npu_initialized_before_config/after_config` 保留实际值；真实设备场景允许为True。
独立 `--run-config` 仍要求配置前后NPU未初始化，避免将设备行为记为CPU证据。
此修复只涉及验证probe；更新时同步拉取对应冻结包，保留旧结果，使用新输出目录重跑。

### 模型加载后读取参数时报内部格式错误

如果错误来自 `dsv41_normal_model_probe.parameter_record` 的 `tensor.detach().cpu()`，
且包含 `copy_ do not support internal format`，查看角色报告的 `model_loaded` 与
`inference_executed`。11月17日回传已完成原生Worker初始化和模型加载，随后在采集参数SHA时失败，
尚未进行KV初始化与推理。

修复后的采集函数只将读取用的detached张量以非原地接口转换为ND格式，再复制到CPU；
保留原模型参数存储、SHA/finite检查及原生/AFD对照。SHA表示逻辑连续元素字节，不含内部padding。
共享helper和默认冻结包已同步，更新后保留旧结果，改用新目录执行原命令。
实际A5读取及后续场景仍待重跑，本地CPU检查不能代替设备验收。

### CompressorV2 的算子库或符号未找到

`aclnnCompressorV2 or aclnnCompressorV2GetWorkspaceSize not in ...` 表示原生
调用未找到对应库或导出符号。此前A5原生入口已为此设置Transformer vendor路径；
旧集中验证入口只继承现有环境，遗漏了显式`--transformer-vendor`选项。
更新入口和同版冻结包后，在第1节命令中传入实际vendor根目录即可，默认示例为
`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer`。
必须在子进程启动、导入torch及算子调用方之前设置两组路径，运行中只修改Python
进程的`LD_LIBRARY_PATH`不等同于重新启动加载器。

参数检查只核对`op_api/lib/libcust_opapi.so`存在，CPU路径不加载该库或执行内核。
预检保留实际环境和原生库身份；库的实际符号及CompressorV2执行仍需A5重跑确认。
本次[检查记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-transformer-vendor-20261117/README.md)
与此前QLI几何修复独立保留，不能将两种失败混为同一原因。

### 首轮 QLI metadata 拒绝 64 个 Indexer 头

11月17日 `rerun3/native-eager-u1.log` 的实际错误为：

```text
ValueError: QLI metadata requires N1=32, N2=1 and logical D=128
```

该次原生Worker、参数读取、KV初始化及公开warmup均已完成；首轮prefill在
`prepare_attn -> build_qli_metadata -> quant_lightning_indexer_metadata` 中失败，
尚未进入模型forward或采样。报告为 `model_loaded=true`、
`attention_kv_initialized=true`、`inference_executed=false`。
已加载的indexer `wq_b.weight=[8192,64]`、`weights_proj.weight=[64,512]`
确认受控配置实际声明64个indexer头；A5当前DSL算子要求32个。
两处数字分别指Attention与Indexer，不能互相替代，也不能通过更改TP来修复：
锁定的原生Indexer在每个TP rank复制完整index heads。

后继冻结包在首次 `hf_overrides` 中声明 `index_n_heads=32`、
`index_head_dim=128`，正常原生及A/F使用同一配置，Attention仍为64头、维度512。
同批DSpark受控配置也同步32个indexer头。生产资格校验仅在显式functional候选内
增加32头声明，保留旧64头候选和其他限制；没有修改原模型文件或上游算子。
参数形状已变化，需重新执行本批原生/AFD对照，不能复用旧轮参数SHA或采样结果。

更新分支后，使用**新的输出目录**执行第1节命令：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
```

提交同时更新源码、默认冻结包和锁；入口文件SHA保持不变，无需在A5重建冻结包。
本次[原始日志与检查记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-qli-geometry-20261117/README.md)
保留独立失败证据。CPU配置检查不执行A5 QLI内核，实际首轮、后续采样和七场景仍待重跑。

### 交付前先在 A3 检查共用脚本

修改本套验证脚本后，先用同一冻结包在A3隔离环境检查共用路径，再交付A5设备验证。
门禁覆盖七场景metadata、命令行参数、正常/异常/超时进程回收、原生与A/F的13种真实配置
（含U2多流开关）、真实ND/NZ参数读取、原生验证模型加载与参数采集；KV规划和分配单独检查。
每轮保留真实退出码、原始日志、来源SHA、清理结果，结束时确认无残留与八卡空闲。

A5入口继续保留真实硬件门禁；A3应被该门禁拒绝，不能伪装成A5绕过检查。
A3的脚本检查不代表A5专用bootstrap、Graph/U2执行或DSpark/PD七场景推理已通过。
本次检查记录见 [A3共用脚本验证](../../openspec/changes/afd-hccl-full-parity/evidence/a3-functional-script-check-20261117/README.md)。

## 4. 源码与实施边界

交付源码见 [准备记录](../../reports/dsv41-a5-functional/preparation.json) 和
[逐文件锁](../../reports/dsv41-a5-functional/bundle-lock.json)。入口解压并运行冻结源码，
各场景共享同一套 146 个生产模块，避免与本地其他开发改动混用。可直接在更新后的仓库执行，
不要求环境具备本地历史证据目录，也无需在 A5 重建验证包。

本轮补齐默认关闭的 `AFD_DSV41_FUNCTIONAL_QUALIFICATION=1` 正常 Worker 组合入口，
保留固定 BF16/dummy/C2/TP1/两请求上限、原生 DSpark、attested PD 等约束。
旧 eager/U1 资格入口和默认拒绝不变；脚本在首 EngineArgs 声明前设置所需值，
不通过构造后的改配置开启 U2/Graph。包含已有 Graph/U2 跨层 PD 准入和 FFN
后台异常向公开 poll/shutdown 传播的修复；未纳入暂停的旧 V4 MTP 改动。

完整检查点首 token 等价仍需与此前
[同 TP4 原生/AFD-F 单层 MoE 对照](DSV41_A5_AFD_MOE_PARITY_ZH.md)的结果一起分析。
该对照不加入本批默认七场景；原生 MoE 已有结果，不必重复原生单独对照。
Engram 专项、完整五种 PD 模型矩阵、代表性拓扑/长运行及性能/服务启停验收继续独立登记，
本批不会提前将这些任务勾选完成。
