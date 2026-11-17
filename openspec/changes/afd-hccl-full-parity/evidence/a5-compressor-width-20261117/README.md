# A5 CompressorV2 最小输入宽度修复

用户上传`be689bf9c63d4da9b744c1f90c0816b3.zip`，其中只有
`native-eager-u1.log`。原始字节及SHA见`native-eager-u1.log.gz`和
`operator-source.json`；`failure-summary.json`提供精简失败字段。

原生模型加载、KV初始化和公开warmup已返回；首轮prefill在原生
`DeepseekV41Compressor.pool_projected -> CompressorV2`执行时报507035、
向量错误341和UB地址越界。请求步骤为空，采样未完成，worker shutdown亦记录失败。
这不是AFD通信失败或已通过的原生场景。

日志的受控模型`hidden_size=512`，compressor的wkv/wgate均为BF16 `[512,512]`。
CANN官方[CompressorV2 README](https://atomgit.com/cann/ops-transformer/blob/8e4bfd80ccb385ee7a9cd24c41ea81df1e3d3e05/experimental/attention/compressor_v2/README.md)
声明H为1024至10240，512对齐，D支持128或512。通过官方raw文件与contents API
核对同一Git blob，并用固定commit的raw文件再次核对SHA；引用身份见`operator-source.json`。
H=512违反公开算子约束已确认；没有从本次日志推断设备故障，也没有确认安装二进制的源码版本。

普通Worker功能候选在首次HF override中改为H=1024，保留Attention D=512、
Index D=128/32头和intermediate=512；原生及A/F声明相同。
权重观察使用正常Worker配置中的模型宽度核对embedding、lm_head和mHC形状。
仅显式functional资格增加固定1024宽候选；旧dummy候选仍只允许512，其他约束保留。
DSpark受控组件的512宽配置保持原状，不把该组件视为原生CompressorV2执行证据。
冻结包显式刷新C2观察helper；未修改vLLM、vLLM-Ascend或算子库。

检查：

- `run_cpu_validation.py red --output-dir .../cpu-red-002`使用修复前的
  `a4b3ab6`资格校验和新测试，428项中36失败、392通过，无skip/error。
- `run_cpu_validation.py green --output-dir .../cpu-green`在既有A3环境的隔离目录
  执行428项CPU测试，全部通过；原生eager/U1和A/F各四种U1/U2、eager/Graph配置
  共9次真实EngineArgs均退出0，实际H=1024/D=512/Index32-D128。各进程NPU初始化前后
  均为false。原始日志、JUnit及配置结果在各自`results.tar.gz`内，CPU源码身份逐文件锁定。
- Windows本地测试56通过、372因缺少torch/vLLM跳过；它们由上述无skip的CPU检查补齐。
- `python -B -S -m unittest tests.e2e.diagnostics.test_dsv41_a5_functional_validation -v`
  4项通过，包含全部7个metadata场景和vendor子进程环境传播。
- Ruff检查、格式检查、`.agents/dsv41/harness.py check`及`guard`通过。
  harness仅验证本地规划一致性和上游保护。

`parent-bundle/`保存修改前的默认冻结包；后继包包含185个Python模块、146个生产模块，
194项原生Git源码身份未变。与父包比较只有资格校验、普通Worker probe、C2权重观察
三个Python文件变化，另有自动生成的manifest/provenance；入口SHA未变。
后继默认包SHA为`35bcf327a8f030dd7fe089737f4446e4410657993ef0c2b0593078fbbe535447`。

实际A5 CompressorV2、首轮/后续采样、Graph/U2和DSpark/PD七场景仍待用户重跑。
继续使用带`--transformer-vendor`的统一入口和全新输出目录；不勾选设备验收任务。
