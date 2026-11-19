# A5 完整权重首批回传：插件注册缺失

输入：`cf6518bc85c34bff9bbf0950003c88bd.zip`，SHA256 为
`1d1c920a75d1ec1b62e5493cfa8ec298fb975e5729589e908a7e0a63159dec11`。
本次仅离线读取压缩包与本仓库源码，没有在 A5/A3 安装、导入运行时或启动设备流程。
原始 JSON 保持字节不变；归档哈希及汇总见 [review.json](review.json)。

| 归档 | 声明配置 | 实际到达阶段 |
| --- | --- | --- |
| `full-eager-u1-1wpm_sg2.tar.gz` | 原生 TP4、卡 0–3、CPU Engram、DSpark/PD 关闭 | 预检失败，没有服务或请求日志 |
| `full-eager-u1-2zhutj6p.tar.gz` | AFD A4/F4、卡 0–7、eager/U1、CPU Engram、DSpark/PD 关闭 | 相同预检失败，没有服务或请求日志 |

两次都使用 Python `/usr/local/python3.11.10/bin/python`，插件仓库 HEAD 为
`d0ae16adf59a016072a968487d57ac577614e19a`，受检的 tracked source status 为空。
25 份 vLLM/Ascend 原生文件均与预期 SHA256 一致；checkpoint 的 config/index
身份和 48 shard、96,085 tensor 的结构检查通过。这里只读取 header，
`tensor_payload_bytes_read=0`，不代表权重 payload 完整性或载权通过。
`device_probe_started=false`，包内没有 `summary.json`、服务日志或 token 对照结果。

两次 `preflight.json` 唯一错误均为：

```text
missing package metadata: vllm-afd-plugin
plugin entry point differs: afd
```

`packages.vllm-afd-plugin=null`；`general_plugins` 中没有 `afd`。同时
`package_roots.afd_plugin` 已指向当前 checkout，因此不能把能定位源码理解为
已经安装插件入口，也不能据此判断部署用了旧 checkout。Ascend 的版本字符串
没有被用作源码一致性结论；25 个受检文件匹配不代表上游整仓版本一致。

根因对应的本地契约：

- [pyproject.toml](../../../../../pyproject.toml) 声明 distribution 为
  `vllm-afd-plugin`，`vllm.general_plugins` 中的 `afd` 为 `afd_plugin:register_afd`。
- [预检器](../../../../../tools/diagnostics/preflight_dsv41_a5_afd.py) 读取
  `importlib.metadata.version()` 和 `entry_points()`；模块位置单独校验。
- [setup.py](../../../../../setup.py) 的 `AFD_BUILD_ASCEND_OPS=0` 禁止扩展编译。
  仅 `AFD_SKIP_ACLNN_BUILD=1` 仍可能进入 pybind 构建，故本轮不使用它代替前者。

指导遗漏了使用当前 Python 注册新 checkout 的准备步骤，现已补充
[安装注册及只读核验](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)。
命令使用 `--no-deps --no-build-isolation --no-index -e .`，沿用已有构建工具，
只注册 AFD，不下载包或切换运行依赖；未放宽原生源码、checkpoint 或入口校验。
该命令尚未在用户 A5 执行，注册成功与随后推理仍待回传。

当前两个目录都不能用作 `--reference-run`，因为还没有成功请求和正常退出的
`summary.json`。下一步先完成注册核验，再重跑原生 TP4；成功后使用新生成的目录
运行 AFD eager/U1。首 token 差异、U1 输出等价及其他完整模型矩阵均未在本包获得证据，
任务勾选保持原状。无需为诊断这两个错误先重跑 U2/Graph/DSpark/PD 场景。
