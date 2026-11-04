# C0 环境准备与只读采集

日期：2026-11-04。当前范围是开始 C0 前的环境准备与证据采集；尚未迁移运行代码。
本页可独立用于环境采集。固定上游基线为 vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` 和 vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`；采集结果用于与这些基线对照。

## A5 已知信息与待收集内容

用户提供镜像 `vllm-ascend-zk5:v2.1`，FastAPI 使用 `0.123.0`。
这两项目前属于用户提供的信息；镜像内实际包、源码、设备和导入行为待采集。
镜像标签可以变化，需要同时记录镜像 ID 和可用的 RepoDigests。

需要收集：

| 信息 | 用途 |
| --- | --- |
| 镜像 ID/digest、实际容器、Python 路径 | 将结果绑定到实际运行环境 |
| vLLM/Ascend/AFD 版本、Requires-Dist、源码 HEAD/工作区 | 与固定 SHA、原始约束和安装包元数据对照 |
| FastAPI/Starlette/Pydantic、Torch/torch-npu、全部包版本、pip check | 区分现有镜像状态、依赖一致性和后续运行测试 |
| npu-smi、驱动/CANN/算子版本、自定义 OPP 路径 | 确定目标平台与需要补充的算子检查 |
| 模型 config.json、权重索引大小/分片数、checkpoint 类型 | 确认 V4.1 Flash、量化与 Engram；不读取权重内容 |
| 计划的 TP/EP/DP、最大上下文、并发、A/F 卡数与主机连接 | 为权重、Engram、KV、workspace 和通信做峰值预算 |

FastAPI 0.123.0 满足锁定 Ascend 的 `<0.124.0`，不满足锁定 vLLM 的
`>=0.133.0,<0.137.0`。因此继续记录 B1；既有镜像可以先做只读盘点，后续导入、
组件和运行结果各自记录。不能仅凭已安装、能够导入或健康端点通过 G1。
不改上游 requirements、METADATA 或已安装文件，也不重装镜像。

## A5 采集命令

在 A5 更新仓库后，从仓库根目录执行以下 Linux shell 命令。
脚本为 [collect_dsv41_environment.py](../../tools/collect_dsv41_environment.py)，
仅需 Python 3.10 或更高版本。先确认镜像和运行容器：

```bash
docker image inspect vllm-ascend-zk5:v2.1 \
  --format 'id={{.Id}} digests={{json .RepoDigests}} os={{.Os}} arch={{.Architecture}}' \
  > a5-image.txt
docker ps --filter ancestor=vllm-ascend-zk5:v2.1 \
  --format '{{.ID}} {{.Image}} {{.Names}}'
```

在实际使用的运行容器、服务 Python 和已配置 CANN 环境中采集。
将下方容器名替换为上一条命令的结果；如果使用虚拟环境，将 `python3` 换成该环境
Python 的绝对路径。`docker exec` 不会自动加载交互式 shell 中的 CANN 设置；若有
专用环境激活方式，应在该现有 shell 内运行脚本。失败信息也应保留。

```bash
AFD_A5_CONTAINER=替换为运行容器名
docker exec -i "$AFD_A5_CONTAINER" python3 -B - \
  --image-ref vllm-ascend-zk5:v2.1 \
  < tools/collect_dsv41_environment.py > a5-container-environment.json
python3 -B tools/collect_dsv41_environment.py > a5-host-environment.json
```

若没有对应的运行容器，先返回镜像信息和宿主机结果，再确定镜像的实际启动方式。
这里不自动创建容器或推理服务。无需导出整个镜像。

知道路径后，在容器采集命令的 `--image-ref` 后追加以下参数；只追加实际存在的路径：

```text
--vllm-source /实际/vllm源码目录
--ascend-source /实际/vllm-ascend源码目录
--afd-source /实际/afd-plugin源码目录
--model-config /实际/checkpoint/config.json
```

脚本仅使用 Python 标准库，读取发行包元数据、版本文件、配置与权重索引，
运行 `pip check`、`npu-smi info` 和只读 Git 命令。Git 命令关闭可选锁和 fsmonitor。
不导入 Torch/vLLM，不执行 NPU 算子，不读取权重文件，不安装包。
子命令返回码和 stderr 独立写入 JSON；脚本退出 0 只表示采集完成。
可先返回 `a5-image.txt`、`a5-container-environment.json` 和 `a5-host-environment.json`，
再补模型及部署参数。未知路径不会阻止第一轮采集。

## A5 权重张量明细补采

已提供过配置和索引摘要的环境，只需补采完整索引及 safetensors 文件头。
更新仓库后，在能读取模型文件的 A5 容器内执行：

```bash
git pull --ff-only
python3 -B tools/collect_dsv41_tensor_metadata.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --output /tmp/dsv41-tensor-metadata-$(date +%Y%m%d-%H%M%S).json.gz
```

命令末行会显示输出路径。返回该 `.json.gz` 文件即可，无需传输权重数据。
脚本仅需 Python 3.10+ 标准库，不导入推理框架、不使用 NPU；只读模型配置、完整
索引以及各分片的 8 字节长度前缀和 JSON header，不读取张量数据区。
报告保留张量全名、dtype、shape、offset 和分片归属；字节数按 offset 差值统计，
不将 FP4 或 packed U8 误当作 BF16。

只接受 `DeepseekV41ForCausalLM`，防止混用 V4 或 DSpark checkpoint。
输出必须位于模型目录之外，且不会覆盖已有文件。缺片、非法 header、索引归属或
总量不一致时，输出 `complete=false` 并退出 1；该报告仍可用于定位需要补采的分片。
config/index 无法读取或架构不符时直接报错。采集成功不代表模型或数值验收通过。

## A3 初次只读探测结果

使用用户提供的 SSH 别名从本机 PowerShell 登录成功。
首次 SSH 指纹保存在本仓库本地 artifacts 中，未修改全局 known_hosts。
当前探测针对 SSH 默认 Python，不能据此排除其他虚拟环境。

| 项目 | 实测结果 |
| --- | --- |
| 系统 | aarch64、Ubuntu 24.04.3、Python 3.12.3 |
| NPU | npu-smi 显示 8 个 NPU 编号，每个 2 个 chip，共 16 个 chip；每 chip HBM 65536 MB |
| 健康与占用 | 本次显示全部 OK；工具未发现运行进程，不代表后续可用性或独占资源 |
| 驱动 | npu-smi / driver version.info 均为 25.5.5 |
| 默认 Python 的核心包 | 未发现 Torch、torch-npu、vLLM、vLLM-Ascend、AFD、FastAPI 发行包 |
| 依赖检查 | pip check 退出 1，报告多项缺失依赖，原始结果保留 |
| CANN/容器工具 | /usr/local/Ascend 下仅看到 driver；PATH 中未发现 docker |
| /dev/shm | 64 MiB；实际运行前需结合进程布局审查 |

初次 `npu-smi info` 退出 127，提示缺少 `libc_sec.so`。
只为该命令设置驱动库搜索路径后退出 0：

```text
env LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver npu-smi info
```

没有修改远端文件、全局环境、安装包或启动服务。A3 后续可用于通用接口、worker
和 HCCL 组件验证，但仍需确认可用软件环境；未执行 NPU 算子、HCCL 通信或模型加载。
完整 checkpoint 能否在这台机器运行，要等实际权重/量化、Engram、KV、workspace
和并行布局预算。A5 算子/数值及系统验收仍需 A5 实测。

原始证据位于本地 `.artifacts/dsv41/c0-preflight-20261104/`：
`a3-environment.json`、`a3-npu-smi.txt`、`a3-npu-smi-driver-path.txt`。
该目录受 Git 忽略；后续分享证据需明确附带文件。

## 初版环境采集工具验证

本地完成 Python 3.10 语法兼容检查、JSON 输出、模型配置哈希/索引摘要、Git 源码
采集、缺失路径处理和文档链接检查，结果通过。夹具配置不是真实 checkpoint。
A3 Python 3.12.3 已成功运行采集工具并生成可解析 JSON；内部 `pip check` 和
未配置库路径的 `npu-smi` 失败如实保留。详细命令和结果见 artifacts 下
`validate_collector.py`、`local-validation.txt` 和 `evidence-manifest.json`。
未运行 AFD 单元/集成测试；本机 Python/WSL 中未找到 Ruff，未安装额外依赖。

## 后续使用这些输入

收到 A5 采集结果后，先核对镜像/源码/包的一致性并归类 pip check 结果，
再明确可运行的导入和组件检查。保留原始依赖冲突与未执行项。
C0 的固定源码接口审查可以独立开展；是否开始运行代码迁移按最新用户指令确定。
当前没有完成任何 C0 整项任务，也没有取得目标 NPU 运行验收结果。
