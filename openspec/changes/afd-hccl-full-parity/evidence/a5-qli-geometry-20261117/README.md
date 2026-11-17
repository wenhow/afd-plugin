# A5 首轮 QLI Indexer 几何失败：2026-11-17

[用户回传日志](native-eager-u1.log.gz)保留原始97,238字节，SHA256为
`dedeeee95529076b866eaf5815b80aacafdae36c743e3691f5c818500ddf5dc6`。
来自`dsv41-a5-functional-20261117-rerun3/native-eager-u1`。
[角色报告](reported-result.json)及[失败摘要](archive-analysis.json)另存。

## 直接原因及证据范围

`model_loaded=true`、`attention_kv_initialized=true`、
`compile_or_warm_up_returned=true`；参数SHA读取也已完成。
真实Scheduler首轮prefill进入`NPUModelRunner.execute_model -> prepare_attn ->
build_qli_metadata -> quant_lightning_indexer_metadata`，当前已安装DSL算子抛出
`QLI metadata requires N1=32, N2=1 and logical D=128`。
`inference_executed=false`，没有forward、采样或A/F对照结果。

实际indexer的`wq_b.weight=[8192,64]`及`weights_proj.weight=[64,512]`
对应64个index heads、维度128；并非只根据普通Attention的64头推断。
锁定Ascend `7e37e29a` 的`attention/dsa_v41.py:build_qli_metadata`
直接以`index_n_heads`和`index_head_dim`构造算子参数；
`models/deepseek_v41/indexer.py:DeepseekV41Indexer`复制全部index heads到各TP rank，
不能通过增大TP将64自动切成32。当前安装的A5 DSL限制来自本次真实异常，
不将本地旧C++ QLI或A3算子能力代替它。

报告`cleanup_failures=[]`、`native_groups_destroyed=true`。只有这一份角色日志，
没有外层退出/全机空闲快照，不能据此宣称整轮清理或七场景通过。

## 修改

正常功能probe首次EngineArgs的`hf_overrides`声明32个indexer头，维度128；
普通Attention仍64头、维度512，C2 `[0,0,2]`及HC、MoE语义保持。
配置记录与首次声明显式保留Indexer两项几何，原生/A/F共用此配置。
资格校验仅在默认关闭的functional候选中增加32头，历史64头仍独立允许；
旧dummy候选继续只允许64头。拒绝非整数与其他几何，没有修改上游或原检查点。

同批DSpark helper也从不可变原件重建A5 successor，indexer统一32头，
避免后续同类错误。通用A3 probe和已归档源/结果均不改写。
准备器只刷新两份相关A5 helper，其他DSpark成员仍来自原冻结包。
新模型参数形状变化，需要重新跑原生/AFD参数及采样对照，旧轮结果不能混用。

[父冻结包](parent-bundle/bundle-lock.json)SHA为
`817d04ee81dc55a1e3d9053dfdd424eeb83df8f1045f09367dec4359d8d320f7`；
后继SHA为`28472209418f029a957f037fd1e8581bf18f34c13b7923db15797c875d0589bb`。

## 检查

[CPU运行器](run_cpu_validation.py)在现有真实依赖venv的隔离目录执行，
不安装共享包、不构造Worker、不执行设备模型。原生环境设置复用既有CANN/ATB。
后继runner拆成[远端代码](cpu_native_runner.py)，仅整理布局；运行产物保留实际冻结源码。

- [RED结果](cpu-red-002/results.tar.gz)：相同新增边界测试386项，36失败、350通过、
  0跳过/错误；失败为旧准入拒绝32头。NPU初始化前后均False。
- [GREEN结果](cpu-green/results.tar.gz)：386通过、0失败/错误/跳过；
  单测源码与冻结记录一致，NPU初始化前后均False。
- [九种首配置](cpu-green/summary.json)：原生eager/U1，A/F各四种eager/Graph、U1/U2，
  全部退出0，实际几何均Attention64、Indexer32/128，NPU初始化前后均False。
- Windows CPU单测56通过、330依赖跳过，见[JUnit](local-unit.xml)；跳过不计通过。
- 七场景stdlib入口、185Python编译、194原生Git对象身份、冻结包成员校验、
  Ruff与规划/上游保护检查见[本地检查](local-validation.json)。

CPU检查没有执行A5 QLI算子。实际首轮、采样、A/F数值和七场景设备验收待重跑。
保留现有任务勾选，不将本轮配置修复记成完整模型、Engram或性能通过。
