# A5 CompressorV2：已安装 custom_transformer，缺少发现路径

日期：2026-11-10。来源：用户回传 `f1a363f791b04b38baef5cafbec28c46.gz`，
SHA256 `6c34c5208dd633188c267c4b377e9323968c99ce4b12288e580b9f27930c9f1e`。
采集目录 `dsv41-opapi.OgEtIk`；采集工具退出码 0 仅表示完成采集。
留存文件中仅 `environment.txt` 去除了 `pip show` 空字段的行尾空格，其余采集内容原样保存。

## 观察与结论

[原始报告](libraries.json) 对每个库使用独立子进程加载并查询符号。
`/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so`
可加载，且 `aclnnCompressorV2`、`aclnnCompressorV2GetWorkspaceSize` 均存在。
内置 `libopapi.so`、`libopapi_transformer.so` 可加载但不包含这两个接口；
裸库名 `libcust_opapi.so` 返回找不到文件。

[环境记录](environment.txt) 显示 `ASCEND_CUSTOM_OPP_PATH` 为空，
`LD_LIBRARY_PATH` 没有 custom_transformer 的 `op_api/lib`。
所以已确认的缺口是已安装 vendor 的发现路径，先补环境并重新启动进程。
不以此前报错中的通用库名推断必须重装 CANN、缺少整个算子包或算子不支持本机。
用户随后列出的 `/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/`
也有 `libcust_opapi.so`；若通过该路径使用，可显式令
`AFD_TRANSFORMER_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer`。
新启动器会验证指定路径的实际库及符号。

## 修正的依据与边界

只读参考：Ascend 锁定对象 `7e37e29a7803d6897a7113088e2b771f83205bc0`：

- `csrc/aclnn_torch_adapter/op_api_common.h` 的 `get_custom_lib_path` 读取
  `ASCEND_CUSTOM_OPP_PATH`，给每个 vendor 根目录追加 `/op_api/lib/`，再加载 `libcust_opapi.so`。
  `g_custom_lib_path` 在 native 初始化时保存此值；环境必须在进程导入前设置。
- `csrc/cmake/scripts/custom/install.sh` 生成的 `bin/set_env.bash` 同样将 vendor 根目录加入
  `ASCEND_CUSTOM_OPP_PATH`，将其 `op_api/lib` 加入 `LD_LIBRARY_PATH`。

上述是锁定源码的契约参考，不声称本轮重新核验了部署环境全部二进制与该提交一致。
启动器采用同样的环境设置方式，保留已有路径；默认 vendor 从当前 `ASCEND_OPP_PATH`
解析，只允许用 `AFD_TRANSFORMER_VENDOR` 显式覆盖，不扫描其他安装或修改上游。
默认原生路径先执行目标绝对库的 ctypes 加载与符号查询，失败即停止，避免反复加载整模型。
日志分别落入 `environment.txt`、`opapi-preflight.json` / `.log`、`native-server.log`。
独立 TP8 activation probe 不新增此 CompressorV2 前置条件。

本回传没有执行算子、NPU 推理或新的请求。它证实符号可用，不能证实 kernel、ABI/算子运行、
首请求成功或完整原生/AFD 等价性。此前 [首请求失败](../a5-20261110-native-ep8-request-001/diagnosis.json)
保留为历史证据；任务 1.10/1.11 继续未完成。

下一轮依照 [主操作说明](../A5_NATIVE_EAGER_ZH.md) 更新并启动，沿用 DP8/TP1/EP8、
CPU Engram shared memory、eager、DSpark off 和同一个小请求。
本地 [启动器验证](launcher-validation.json) 仅覆盖 CPU stub 的环境、退出和日志行为。
