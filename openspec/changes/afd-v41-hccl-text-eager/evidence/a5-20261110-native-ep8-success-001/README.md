# A5 原生 DP8/TP1/EP8：完整冷加载与首次文本请求通过

日期：2026-11-10。用户回传 `6c31728c57fb470d8d846d7fd808dd17.gz`，
SHA256 `cad58629c795b4573f419ed63bfe3f3a724ac9088ff7ca02b726893818f7ed04`。
原始结果目录为 `dsv41-native-dp8-ep8.oZJo2T`；[完整原包](raw-results.tar.gz) 原样留存，
含服务全日志、启动器日志、请求/响应和四份设备快照。
[摘要](summary.json) 保存每个原始文件的 SHA256；`environment.txt` 的可读副本仅去除行尾空格。

## 已验证的结果

8 rank 均完成完整 checkpoint 冷加载，8 个 API worker 报告 startup complete。
服务日志 LF 行号 1392 为 `GET /health` 200；1403 为 `POST /v1/completions` 200，
见 [关键日志摘录](server-excerpts.txt)。没有在 SIGINT 前发现 ERROR、traceback 或原先的 CompressorV2 符号解析异常。

[请求](request.json)：`The capital of France is`，temperature=0、seed=0、max_tokens=32、
logprobs=5、return_token_ids=true、stream=false。
[响应](response.json)：文本 ` Paris.`，finish_reason=`stop`，token IDs `[11111,16,1]`，
输入 5 token、输出 3 token（含结束 token），请求退出码 0。
返回了 token logprobs，但本轮没有同条件参考输出逐项对比，不能据此宣称数值等价或准确率通过。

此次不是符号查询替代推理：完整原生服务实际返回了完成响应，先前首请求的失败没有重现。
证据只覆盖这一个短 greedy 请求，未覆盖长输入、连续/并发、取消/回收、DSpark、AFD 或性能。

## 实际环境与配置

- afd-plugin：`e7cfe618aaba693f40a587424c58a90e9d8a790e`；工作区状态仅有 `?? results/`。
- Ascend 实际 HEAD：`4ec6dc7722ec4f7f8005a86570238ac4377c5d43`，其 status 无输出；
  包元数据仍为 `0.19.1rc2.dev637+g2db8549c5`。项目锁定为 `7e37e29a…`，未改锁或视为完全匹配。
- vLLM 包元数据 `0.30.0+empty`；Python 3.11.10、torch 2.10.0、torch_npu 2.10.0.post4、
  FastAPI 0.123.10。包元数据不等于核验了 vLLM 全量源文件；声明依赖差异继续保留。
- Ascend950DT，0–7 卡；DP8 / TP1 / EP8，PP/PCP/DCP=1；V2、eager、DSpark off、DBO off。
- `deepseek_v4_fp8`、safetensors、text-only；长度/批 token 上限 1024、每 DP engine 最大 1 条序列。
- Engram `cpu_offload=true`、`dp_shared_memory=true`、`embedding_across_dp=false`；`/dev/shm=512G`。
- 仅启用 Ascend 插件列表，`AFD_DSV41_MXFP8_SCALE_PATCH=0`；已安装的 AFD entrypoint 不代表本次加载了它。
- config/index SHA256 与既定值一致，48 个 shard 均存在；见 [预检查](preflight.json)。
  这不是 48 个权重 payload 的全量 hash 校验。

本次显式选择 `/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer`，
其 vendor 根目录和 `op_api/lib` 分别加入 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`。
[opapi 预检查](opapi-preflight.json) 的库加载和两个 CompressorV2 符号均通过，
与此前 [库诊断](../a5-20261110-opapi-001/README.md) 一致。
服务日志还记录了 `compressor_v2` 扩展构建，以及 60 秒 broadcast 等待信息；
之后请求成功。没有计时字段支撑 TTFT、TPOT 或性能结论，也不把该等待信息视为本轮请求失败。

## 内存与退出边界

模型加载报告为每 rank 44.29 GiB，KV 预算为每 rank 34.11–34.12 GiB。
请求后的 `npu-smi` HBM 设备总占用显示 89653–92419 MB/卡，容量显示 98304 MB/卡；
这是工具原始单位的离散快照，包含模型以外的进程/系统占用，不是模型独占或实测连续峰值。
原包保存了 before、ready、after-request 和 after 四份快照，摘要逐卡列值。

成功响应之后，日志行 1404 起于 19:16:08（Asia/Shanghai）记录 SIGINT；
`launcher-exit-code.txt` 为 130，没有 `server-exit-code.txt`。
首个 ERROR 在行 1496，后续有 `EngineDeadError`、API process 退出错误与
8 个 shared_memory 对象待清理的 tracker 警告。
`npu-after.log` 仍显示多个本次 worker，快照采于退出/清理阶段；
没有延迟复查，无法判断之后是否全部释放，也不能据此认定永久泄漏。
因此首次请求 smoke 通过，退出回收和重启仍待验证；不能把退出阶段错误倒推为 HTTP 请求失败。

## 记录与下一步

本地使用标准库解包、JSON/退出码、日志顺序、8 rank 和设备快照一致性核对，全部通过；
本地未执行 NPU、请求或性能实验，实际推理证据来自用户本次 A5 回传。
主说明已更新为首次原生推理 smoke 通过；tasks 1.10 记录本次实际环境的通过项，
锁定源码对完整资格、更多输入控制组与退出回收不因单次成功自动完成，AFD 1.11 仍待验收。
后续保持当前可用配置，补连续/并发、退出后设备复查与新进程重启，
再建立与 AFD 布局、Engram 放置匹配的控制组。
