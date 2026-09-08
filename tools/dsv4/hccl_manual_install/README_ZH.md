# DeepSeek-V4 AFD 第一期 CANN 9.0.0 手工安装包

该目录用于生成可复制到 A5 或双机/多机环境的安装脚本包。默认生成轻量包，
包内包含安装脚本、一期验证指导书、固定版本清单和 afd-plugin 一期补丁，不包含
三个源码仓库、模型或 Python wheel。目标机安装时重新下载：

- vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；
- vLLM-Ascend `3da28f9414583d2d0b672a8f06d1fae142404bda` 及递归 submodule；
- afd-plugin 分支基线提交 `9db1fb981f5262986e7ea05938e9c6ed57b5a885`。

脚本随后校验并应用包内补丁，将 afd-plugin 恢复为
`manifest/versions.env` 记录的精确目标源码树。下载提交、目标 commit/tree、补丁
SHA256 和包内文件 SHA256 均记录在 `manifest/` 中。完整硬件矩阵和证据回传步骤见
包根目录 `PHASE1_A5_MULTI_NODE_VALIDATION_GUIDE_ZH.md`。

## 0. 从交付 ZIP 开始

如果收到的是 `dsv4-afd-hccl-install-delivery-*.zip`，ZIP 只是指导书和安装包的
外层容器；其中的 `dsv4-afd-hccl-manual-install-slim-*.tar.gz` 才是需要在
目标机解开的实际安装脚本包。按以下顺序操作：

```bash
unzip dsv4-afd-hccl-install-delivery-*.zip
cd dsv4-afd-hccl-install-delivery-*
sha256sum -c SHA256SUMS
sha256sum -c dsv4-afd-hccl-manual-install-slim-*.tar.gz.sha256
tar -xzf dsv4-afd-hccl-manual-install-slim-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-*
vi config.env
bash bin/install_all.sh
```

文件名中的 `*` 匹配构建时间戳。`slim` 表示包内没有完整源码、模型或 Python
wheel；`bin/02_prepare_sources.sh` 会在目标机下载并校验固定源码。

## 1. 生成轻量包

在开发仓库根目录执行：

```bash
bash tools/dsv4/hccl_manual_install/build_bundle.sh /mnt/workspace/artifacts
```

生成物名称为：

```text
dsv4-afd-hccl-manual-install-slim-YYYYmmdd_HHMMSS.tar.gz
dsv4-afd-hccl-manual-install-slim-YYYYmmdd_HHMMSS.tar.gz.sha256
```

已经按 `/mnt/workspace/delivery/config.env.example` 安装过的两台 A3，使用复用
profile 生成专用包：

```bash
CONFIG_PROFILE=dual-a3-reuse \
  bash tools/dsv4/hccl_manual_install/build_bundle.sh /mnt/workspace/delivery
```

文件名包含 `slim-dual-a3-reuse`。该 profile 固定复用以下已有资源，不重装 Python
依赖，不重建 vLLM/vLLM-Ascend：

- `/data/z00569729/code/.venvs/afd-v023-vllm-cann`；
- 两个已固定提交的上游源码目录；
- `/usr/local/Ascend/cann-9.0.0` 和既有模型；
- 提交 `2164240b31efc8605bf84cc45afc628996669554` 的旧 `afd-plugin` 只读种子。

包内增量 Git bundle 会从旧种子创建独立的
`/data/z00569729/code/afd-plugin-phase1-a5`，不会修改或切换旧仓库。安装仍会严格
核验依赖版本、三个源码提交/工作树、custom ops、导入路径和 NPU 可见性；不匹配时
停止，不会静默覆盖。

## 2. 目标机前提

- AArch64、满足所选 A/F 拓扑数量的可用 Ascend NPU；
- CANN 9.0.0 和 Python 3.12 已安装；
- DeepSeek-V4-Flash W8A8 模型已放到目标机；
- 可访问配置中的三个 Git 地址、vLLM-Ascend submodule 地址和 Python 包源；
- 已安装 `git`、`tar`、`curl`、`iproute2`/`iproute` 等基础工具。

如果目标环境使用内部 Git 镜像，可在 `config.env` 中改写三个 `*_GIT_URL`，
但镜像必须包含清单指定的提交。

## 3. 解包和配置

先校验并解包：

```bash
sha256sum -c dsv4-afd-hccl-manual-install-slim-*.tar.gz.sha256
tar -xzf dsv4-afd-hccl-manual-install-slim-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-*
vi config.env
```

必须修改：

- `CANN_ROOT`：目标机唯一使用的 CANN 9.0.0；
- `MODEL_PATH`：DeepSeek-V4-Flash W8A8 模型路径；
- `PYTHON_BIN`：Python 3.12；
- `SOC_VERSION`：目标机真实 SoC；
- `NIC_NAME` 和 `HCCL_IF_IP`；
- 安装、源码、日志目录；
- 必要时修改 Git 和 pip 镜像地址。

轻量包保持 `USE_BUNDLED_SOURCES="0"`。不要把验证机 IP 复制到其他机器，
也不要在已经 source 其他 CANN 版本的 shell 中继续安装。

## 4. 校验和安装

```bash
sha256sum -c manifest/SHA256SUMS
bash bin/00_print_config.sh
bash bin/01_preflight.sh
bash bin/02_prepare_sources.sh
bash bin/03_create_venv.sh
bash bin/04_install_python_deps.sh
bash bin/05_install_stack.sh
bash bin/06_verify_install.sh
```

也可以一次执行安装阶段：

```bash
bash bin/install_all.sh
```

`02_prepare_sources.sh` 会克隆固定提交、初始化 vLLM-Ascend submodule、校验
afd-plugin 补丁并比对最终 Git tree。任一版本不匹配都会停止。

这些脚本默认拒绝复用非空源码目录或已有 venv。确认目录内容正确后，分别设置
`REUSE_SOURCES=1` 或 `REUSE_VENV=1`。

`dual-a3-reuse` 包已设置 `REUSE_SOURCES=1`、`REUSE_VENV=1`、
`INSTALL_PYTHON_DEPS=0` 和 `INSTALL_UPSTREAM_STACK=0`。已按上述双机配置执行过的
节点无需重新安装 env；只安装新版 afd-plugin editable 路径。

双机 PD common 模板也已预置，但 `NATIVE_GOLDEN_PATH` 指向的 A5 native control
结果不随包分发；完成 A5 control 后需把该文件放到两台机器的预置路径，或只修改该
路径项。

## 5. Python wheel 离线安装

轻量包的源码阶段需要访问 Git；`OFFLINE=1` 只控制 Python wheel 安装。先在
同架构、同 Python 版本的机器上准备完整 wheelhouse，再配置：

```bash
OFFLINE="1"
WHEELHOUSE="/path/to/wheelhouse"
```

脚本会使用 `--no-index --find-links`，缺少任何直接或间接依赖时立即失败。

目标机完全不能下载源码时，可以显式生成完整源码包：

```bash
INCLUDE_SOURCES=1 \
  bash tools/dsv4/hccl_manual_install/build_bundle.sh /mnt/workspace/artifacts
```

完整包名称包含 `with-sources`，体积会明显增大。要做到完全离线，还需通过
`INCLUDE_WHEELHOUSE=/path/to/wheels` 同时加入完整 wheelhouse。

## 6. 启动和停止

默认配置启动一期最大组合 A8F8 Graph/U2、graph draft MTP N=3：

```bash
bash bin/07_start.sh
bash bin/08_status.sh
bash bin/10_smoke_request.sh
bash bin/09_stop.sh
```

切换到最小定位组合 eager/U1 + MTP N=1 时修改：

```bash
EXECUTION_MODE="eager"
U_BATCHES="1"
ENABLE_MTP="1"
MTP_NUM_SPECULATIVE_TOKENS="1"
MTP_DRAFT_EXECUTION="eager"
```

模型仍使用一个 MTP layer；每轮 speculative token 数支持 `N=1..3`。

## 7. 运行边界

- HCCL Connector 不需要 `pip install hccl`；
- HCCL-only 不构建 afd-plugin CAMP2P custom ops；
- vLLM-Ascend `custom_transformer` ops 仍必须构建和 source；
- TP1 支持 eager/Graph、U1/U2 和 MTP N=1..3；
- `P2pHcclAFDConnector` 支持 `A=kF`、`A=F`、`F=kA` 的整数比例；
- 非整数 A/F、U3、一期范围外并行和正式性能结论仍不在本包验收范围；
- 业务流量只进入 Attention API；
- 启动成功必须同时满足 Attention health 和全部 FFN connector loop ready。
