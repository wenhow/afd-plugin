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

2026-11-18 更新：vendor 路径准备前置后，原生 eager/U1 诊断已完成六轮执行、
取消/追加请求及正常清理，十八次路由调用全部返回。该次仅运行原生诊断场景；
下一步执行完整七场景，见下方“原生 vendor 前置诊断通过”记录。

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
`5732cc4eb3709d660b51c778d236b46dedcc5347bdc110cd2d3d3931e1cfbe30`。
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

### CompressorV2 首轮执行报 UB 越界

11月17日后续上传的 `be689bf9c63d4da9b744c1f90c0816b3.zip` 中，
原生场景已完成模型加载、KV初始化和warmup，首轮prefill进入
`DeepseekV41Compressor.pool_projected -> CompressorV2` 后报
`507035`、`The address for VEC to access UB is out of bounds`。
这次没有完成采样；`cleanup_failures`也非空，不能记为通过。

日志中的compressor权重为`[512,512]`，输入模型宽度`hidden_size=512`。
[CANN CompressorV2约束](https://atomgit.com/cann/ops-transformer/blob/8e4bfd80ccb385ee7a9cd24c41ea81df1e3d3e05/experimental/attention/compressor_v2/README.md)
要求H为1024至10240且按512对齐，D为128或512；原最小模型的H越过了下限。
本轮普通Worker原生/A/F场景在首次配置声明时统一使用`hidden_size=1024`，
保留Attention `head_dim=512`、Indexer `index_head_dim=128`及32个头，
MoE intermediate仍为512。权重观察按实际模型宽度核对embedding、lm_head与mHC形状。
既有受控DSpark组件继续使用独立的512宽配置，不据此宣称其执行了原生CompressorV2。

拉取分支后，继续使用第1节带`--transformer-vendor`的命令，换一个全新的输出目录。
提交同时更新默认冻结包，入口SHA保持不变；无需在A5手工重建归档。
[本次失败与CPU检查记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-compressor-width-20261117/README.md)
单独保留。H=512违反算子约束已确认；本次UB越界是否消除及后续七场景功能仍需A5重跑。

### H=1024 后仍报异步向量核异常

新上传的`aab351a8beb94dd0931099cc8103b0c2.zip`确认H=1024已生效，
但原生首轮仍未完成。日志只有`[Bisheng] Launch kernel failure! ret 507035`，
Python栈停在`torch.ops.vllm.dsa_v41_forward`，随后报stream/event的向量核异常；
运行时明确提示异步栈可能不准确。此日志不能确认实际失败的是CompressorV2还是其他算子。
上一轮修正了H的公开约束违规，没有证明首轮设备异常已解决。

先用同步诊断模式，只运行原生eager/U1：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /mnt/workspace/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --output-dir /mnt/workspace/tmp/dsv41-a5-native-sync-20261117 \
  --diagnose-native \
  --run
```

模型路径沿用实际环境，输出目录必须全新。该参数在启动子进程前设置
`ASCEND_LAUNCH_BLOCKING=1`，只使用第一个设备执行`native-eager-u1`，
保留原生预检、真实Worker及已有清理；不启动A/F、Graph/U2或DSpark场景。
父进程环境不改写，后续正常七场景命令移除`--diagnose-native`即可。
`summary.json`的`diagnostic_only=true`，即使原生运行成功也只标记为诊断，
不会记录七场景通过或原生/AFD等价；Worker报告记录实际继承的同步开关。

回传新目录中的`native-eager-u1.log`、`summary.json`和`native-runtime.json`，
用同步异常栈定位实际失败算子，再决定具体修复。
[本次失败与检查记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-native-sync-debug-20261117/README.md)
与此前H下限、QLI头数和vendor路径问题分别保留。当前首轮及七场景设备功能仍未通过。

### A5 MoE 路由同步越界

`c4416cd61d6b4c05bd80ee69c0ea2662.zip`及补充summary确认同步开关已生效，
原生预检194份源码匹配；首轮实际失败于`_C_ascend.moe_gating_top_k_hash`，
报507035/341 UB越界。gate权重为`[8,1024]`，说明受控模型仅有8个专家；
名称中的hash不代表启用了模型Hash MoE，当前`num_hash_layers=0`。
模型加载、KV和warmup已完成，但尚未完成推理；Worker shutdown失败，八卡最终空闲。

锁定A5内核把8专家补齐到32，而部分UB读取使用完整64个FP32元素的向量。
这是源码分析发现的小形状风险；预检核对Python源码，未证明已安装C++内核的构建来源，
因此仍需设备重跑确认具体成因与修复效果。普通Worker原生/A/F首次声明统一改为
`n_routed_experts=64`，保留top-k=2、H1024和Index32/D128，并记录实际专家数。
functional准入仅增加固定64专家候选；历史8专家及独立DSpark组件保留。

拉取分支后，继续使用上方`--diagnose-native`命令、实际模型路径和全新输出目录，
建议目录名`dsv41-a5-native-sync-experts64-20261117`。先检查原生生命周期和清理结果；
原生通过后，再移除`--diagnose-native`并另建目录运行完整七场景。
提交包含更新后的默认冻结包，无需在A5重建。仍回传`native-eager-u1.log`、
`summary.json`和`native-runtime.json`。

[本轮证据](../../openspec/changes/afd-hccl-full-parity/evidence/a5-routing-width-20261117/README.md)
记录472项真实依赖CPU测试、9组EngineArgs配置及本地/A3各6项CLI检查通过；
CPU过程未初始化NPU。A5越界消除、采样及原生/AFD等价仍待验证。

### 64 专家仍失败：先隔离实际路由算子

11月18日上传的`59a4b91ce20c4c3aa755523e918e496e.zip`确认
H1024、64专家、top-k2和同步开关全部生效，但首轮仍在
`_C_ascend.moe_gating_top_k_hash`报507035/341 UB越界，Worker shutdown也失败。
上一轮64专家调整没有解决问题；目前仍未完成首轮推理。
源码复核还发现编译入口包含同目录的regbase内核，而非此前分析的arch35子目录同名文件；
实际安装二进制的来源尚未核对，不能继续把静态缓冲区推断当成设备根因。

先运行新增的单算子诊断，使用全新目录：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --preflight-timeout 900 \
  --output-dir results/dsv41-a5-router-20261118 \
  --diagnose-router \
  --run
```

该模式保留原生身份预检与前后八卡空闲检查，以六个独立子进程测试：
64专家/top-k2/FP32，384专家/top-k2/FP32，384专家/top-k6/FP32，
64专家/top-k2/BF16，384专家/top-k6/BF16，以及384专家/top-k6/FP32的vision-bias对照。
各用8行固定输入，保持原生sqrtsoftplus评分和权重归一化；最后一项仅为额外算子对照，
不代表启用或验收视觉模型。`--model`仍为入口必选参数，但本模式不加载Worker、模型或权重，
不建立通信组，也不启动普通七场景。

每项记录实际schema、输入文件及SHA、CPU参考ID/权重比较、失败阶段和同步栈，
以及进程实际加载的opapi/tiling/扩展库路径和SHA。失败项退出后，确认所属进程已回收、
八卡空闲，再继续下一独立项；不会在已损坏的NPU上下文中继续测试。
有失败或缺少有效算子结果时，入口最终非零退出，但保留所有已收集结果。
回传新目录的`summary.json`、`native-runtime.json`、`router-*.log`及各
`router-*/result.json`；也可直接压缩整个新目录。

[本轮记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-router-isolation-20261118/README.md)
包含本地与A3各10项CPU检查，未执行NPU算子。146份生产模块和194份原生身份不变。
本次交付用于区分形状、dtype、top-k和实际算子库问题，尚未解决或验收A5启动。

### 六项单算子通过：保存原生失败时的真实输入

11月18日上传的`ee5a52b081874d3ba24a851cd876b18a.zip`中，六个独立算子均成功执行，
路由ID与CPU参考完全一致，FP32权重最大绝对误差为`1.1920928955078125e-7`，
BF16两项误差均为0；六项加载的扩展及opapi库路径/SHA一致，原生194份Python源码匹配。
按用户要求，本次分析忽略NPU占用提示。原始`summary.json`仍保留`success=false`及
`npu_acceptance=failed`，其占用观察不能覆盖六项实际算子结果，也不证明原生启动已修复。
普通随机输入下64专家可以运行，下一步检查原生模型实际路由输入及执行上下文。

更新后运行原生诊断，使用全新输出目录：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --preflight-timeout 900 \
  --output-dir results/dsv41-a5-native-inputs-20261118 \
  --diagnose-native \
  --ignore-npu-occupancy \
  --run
```

`--diagnose-native`现在通过线程局部的Python调用观察器，保存最多64次原生路由调用的CPU输入、
shape/stride/storage_offset、dtype、有限值统计及标量参数；记录实际加载库路径/SHA。
算子仍接收原始参数，输入不会替换为随机对照值。此模式有同步读回和落盘开销，仅用于定位。
若Worker失败且所属子进程已回收，父进程从失败调用的完整记录中选择有限值输入，
启动全新进程重放；恢复逻辑值、形状、步长和存储偏移，每张量分配上限为1,048,576个元素。
记录不完整、含非有限值或没有捕获失败路由时，明确登记跳过原因；不会在损坏的上下文重试。
如果读回输入就失败，需结合保存阶段和原异常判断是否为此前算子的错误。
重放成功仅表明该输入在独立上下文可执行，不代表原生模型或七场景通过；原生失败仍保持失败。

`--ignore-npu-occupancy`仅接受诊断模式，保留全部占用观察并在summary记录排除选择，
继续检查所属进程回收、预检和真实算子/Worker结果。默认七场景仍要求原有空闲检查。
请压缩整个新输出目录，尤其保留`native-eager-u1/router-inputs/*.pt`及同名JSON、
`native-eager-u1/result.json`、`router-actual-replay/`（若生成）、各日志和`summary.json`。

[本轮证据与CPU检查](../../openspec/changes/afd-hccl-full-parity/evidence/a5-router-real-inputs-20261118/README.md)
保留原始上传和独立判读。实际A5输入捕获/重放、原生首轮及七场景验收仍待环境运行。

### 真实输入重放通过：提前准备已安装算子的 vendor 路径

新上传的`71469d06557e4671a2dc7517df967c36.zip`及
`d2b863b33663403fb93ce9b9b3963d4a.zip`确认真实输入保存与独立重放均已执行。
原生首轮仍在hash router报507035/341，Worker shutdown失败；所属子进程最终已回收。
真实logits为FP32 `[8,64]`，bias为FP32 `[64]`，均连续、偏移为0且没有NaN/Inf。
同一输入和标量参数在新进程中成功执行，ID与CPU参考完全一致，权重最大误差
`5.960464477539063e-8`。原始重放报告没有数值参考验收，本轮独立CPU核对另行记录。
这说明不能继续把错误归因于这份输入的逻辑值、普通步长或64专家形状。

两进程实际加载库仍有差异：原生额外加载`opp/vendors/customize/op_api/lib/libcust_opapi.so`。
它是否参与了失败路由调用尚未证实。锁定扩展的`op_api_common.h`在库导入时就把
`ASCEND_CUSTOM_OPP_PATH`存入静态搜索表；普通Worker的A5 bootstrap原先在配置和设备初始化后。
本轮在任何torch/vLLM运行时导入前，把当前已安装vLLM-Ascend自带的custom_transformer目录
放到搜索路径首位，保留外部Transformer及其他原有目录，不修改共享安装或原生算子。
准备报告记录前后路径、库SHA以及扩展是否已经提前导入。该时序调整的设备效果仍待重跑。

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --preflight-timeout 900 \
  --output-dir results/dsv41-a5-native-vendor-first-20261118 \
  --diagnose-native \
  --ignore-npu-occupancy \
  --run
```

仍使用全新目录并回传整个目录（含PT输入）。首个路由调用前及重放调用前增加已加载库、
两个路由ACLNN符号的导出来源，以及inference/grad/deterministic上下文记录。
符号探针只查看已经加载的库，不加载新算子库、不调用算子，也不能证明C++缓存指针的最终绑定。
不忽略原生失败；只有原生生命周期和清理实际通过后，才继续完整七场景。
[本轮记录与CPU核对](../../openspec/changes/afd-hccl-full-parity/evidence/a5-router-import-order-20261118/README.md)
与之前失败和对照分别保留，A5启动及七场景验收仍待确认。

### 原生 vendor 前置诊断通过：继续完整七场景

`a095f8a7ad73413a9fa98abd3eceaf49.zip`及
`e19a3975e09d411a838711d3b1e7c245.zip`确认：冻结包
`d0a32a80a5737f04b968e176cc61f7afa3af444658ecdfc93f6f937b359da521`
在A5原生 eager/U1 完成六轮执行和采样，十八次路由调用全部返回；
A/B两轮后取消A、B结束、追加C并结束，Worker/通信组清理通过，进程退出0且无所属残留。
首次路由输入PT的SHA与此前失败完全相同，前置vendor准备发生在扩展导入前。
该调整后的原生诊断路径现已跑通，原始summary仍为`npu_acceptance=diagnostic_only`。

成功进程仍加载customize vendor库，不能把此前错误解释为“加载该库本身就会失败”。
本次没有原生/AFD比较，也没有运行U2、Graph或DSpark/PD场景；完整七场景仍待验证。
按下面命令继续，使用全新目录，正常模式不带`--diagnose-native`或`--ignore-npu-occupancy`：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --preflight-timeout 900 \
  --output-dir results/dsv41-a5-functional-vendor-first-20261118 \
  --run
```

回传整个输出目录的压缩包，包含snapshot、summary、各角色结果、日志和原始hidden。
七场景通过仍要求`success=true`、`npu_acceptance=component_suite_passed`及七项完整结果。
[本次原始上传与独立核对](../../openspec/changes/afd-hccl-full-parity/evidence/a5-native-vendor-first-pass-20261118/README.md)
记录源码身份、真实输入、六轮输出及清理；完整检查点/Engram/性能范围保持待验收。

### AFD U1 通过，U2 首轮提示源 RoPE 未初始化

`8c282423171142818bdacce74b45ed6a.zip`为U1 Attention，
`6c4d6f688c8f4106a85897c3ff665742.zip`为U1 FFN和pair汇总：A/F均成功退出，
六轮采样、取消A、B结束及追加C通过。独立对照此前保存的原生基线，
A58+F21共79份参数身份和A/B/C输出全部一致；该结论限定于受控BF16 dummy场景。

`89ceeac6f8f74cfcabe3aeeb86948d81.zip`的U2首轮在Attention的C2 metadata构建时报
`Enable device metadata before building compressor metadata`。A/F握手已完成，但尚无计算消息。
原因是AFD新建的子批次builder漏掉原生`prepare_source_rope()`准备；
FFN随后报peer关闭是连带失败。修复对每个新建V4.1 child builder执行原生源表准备，
不在eager中开启异步metadata，不修改上游或算子。新增三项CPU回归先失败后通过，
相关290项CPU契约和17项入口检查通过；实际A5 U2修复效果仍待验证。

更新后用全新目录重跑完整七场景，并回传整个目录：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030

python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 \
  --preflight-timeout 900 \
  --output-dir results/dsv41-a5-functional-u2-rope-20261118 \
  --run
```

本轮局部上传没有整套summary/runtime预检，U1独立对照不补写原始equivalence字段。
下一轮保留summary、snapshot和所有角色/进程结果；新冻结包仅更新一份生产初始化代码，
七场景、U2/Graph/DSpark/PD以及训练检查点完整验收仍按真实结果判定。
[本次证据、源码审查与CPU记录](../../openspec/changes/afd-hccl-full-parity/evidence/a5-u1-pass-u2-source-rope-20261118/README.md)。

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


### 2026-11-18???????DSpark ?????????

[??????????????](../../openspec/changes/afd-hccl-full-parity/evidence/a5-dspark-sender-retirement-20261118/README.md)
???????AFD?????0??AFD?79????????????????
???????BF16 dummy???DSpark?????????P?????
`corrupted double-linked list`???-6??????????????

????DSpark?????drain???????????join???????TE/?????
?????faulthandler????????????????????????????A5???
?? `--dspark-only` ??????????????????????Graph/U2???
????/????????? `dspark_pd_components_passed`???????????
???????????vendor?????

```bash
python -B tests/npu/dsv41_a5_functional_validation.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --devices 0,1,2,3 --preflight-timeout 900 --dspark-only \
  --output-dir results/dsv41-a5-dspark-sender-20261118 --run
```

???????SIGABRT?????????log??faulthandler????
??/??????????????????Engram???????????????????
