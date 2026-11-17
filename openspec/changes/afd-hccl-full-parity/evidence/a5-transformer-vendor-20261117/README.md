# A5 集中入口的 Transformer vendor：2026-11-17

用户提供的错误为`aclnnCompressorV2 or aclnnCompressorV2GetWorkspaceSize not in
libopapi.so ...`，并指出现有vendor目录。此次没有完整角色日志，不能进一步确认
是库路径、依赖库还是导出符号问题，也不能从该错误判断模型已执行到哪一层。

此前`tools/diagnostics/run_dsv41_a5_native_eager.sh`、
`tools/diagnostics/run_dsv41_a5_moe_parity.py`已在启动子进程前设置该vendor的两组路径。
当前集中入口只继承外部环境，缺少显式`--transformer-vendor`选项。

## 修改与范围

入口新增可选vendor参数，校验`op_api/lib/libcust_opapi.so`存在，然后构造本次
子进程环境：vendor根目录前置`ASCEND_CUSTOM_OPP_PATH`，`op_api/lib`前置
`LD_LIBRARY_PATH`。原有值保留，预检、原生、A/F与嵌套DSpark角色都继承；
父进程环境不变。动态库搜索路径在新进程启动前生效，没有通过运行中修改加载器配置补救。
省略参数时保持现有环境继承。实际选择写入`summary.json`的`transformer_vendor`
及`runtime_environment`，现有原生预检继续记录继承环境和库身份。

路径检查不等于符号/算子验收。没有安装、覆盖或加载A5算子库，没有改上游或生产模块。
指导默认命令新增用户给出的vendor目录，并同步入口SHA及冻结包，避免入口/锁不一致。
[父冻结包](parent-bundle/bundle-lock.json)保留上一轮QLI修复原件。

## 验证

执行`python -B -S tests/e2e/diagnostics/test_dsv41_a5_functional_validation.py -v`，
[四项检查通过](cli-validation.log.gz)：真实新Python子进程得到路径且父环境不变；
显式vendor的七场景metadata；未显式vendor时保留继承值；缺库时在创建结果目录前拒绝。
CLI测试的合成库文件仅用于检查路径存在，未加载或模拟设备算子。

185Python编译、194原生Git身份、冻结包成员及146生产身份保持、Ruff与本地
规划/上游保护结果见[本地检查](local-validation.json)。A5库依赖、导出符号和
CompressorV2内核及七场景设备执行仍待用户更新后重跑。任务勾选保持原状。
