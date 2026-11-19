# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def camp2p_config(
    *,
    num_ubatches: int = 1,
    n_shared_experts: int = 0,
    extra_config=None,
):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": extra_config or {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=n_shared_experts,
            ),
        ),
    )


def npu_parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "enable_dbo": False,
        "enable_eplb": False,
        "enable_elastic_ep": False,
        "enable_fault_tolerance": False,
        "use_ubatching": False,
        "num_ubatches": 1,
        "ubatch_size": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "use_sequence_parallel_moe": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def npu_vllm_config(
    *,
    role="attention",
    connector="CAMP2pAFDConnector",
    extra_config=None,
    use_mla=False,
    use_v2_model_runner=True,
    cudagraph_mode="FULL",
    speculative_config=None,
    model_type="deepseek_v2",
    architectures=None,
    engram_config=None,
    kv_transfer_config=None,
    **parallel_overrides,
):
    async_dp = bool(parallel_overrides.pop("async_dp", False))
    compute_gate_on_attention = bool(
        parallel_overrides.pop("compute_gate_on_attention", False),
    )
    num_attention_ranks = int(parallel_overrides.pop("num_attention_ranks", 1))
    num_ffn_ranks = int(parallel_overrides.pop("num_ffn_ranks", 1))
    text_config = SimpleNamespace(model_type=model_type, architectures=architectures)
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        additional_config={
            "afd": {
                "role": role,
                "connector": connector,
                "async": async_dp,
                "compute_gate_on_attention": compute_gate_on_attention,
                "num_attention_ranks": num_attention_ranks,
                "num_ffn_ranks": num_ffn_ranks,
                "connector_extra_config": extra_config or {},
            },
        },
        parallel_config=npu_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_config=text_config,
            hf_text_config=text_config,
            use_mla=use_mla,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                name=cudagraph_mode,
                has_full_cudagraphs=lambda: (
                    cudagraph_mode in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
                ),
            ),
            fast_moe_cold_start=False,
        ),
        speculative_config=speculative_config,
        engram_config=engram_config,
        kv_transfer_config=kv_transfer_config,
        weight_transfer_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )


def hccl_vllm_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    extra_config=None,
    tensor_parallel_size=1,
    dspark=False,
    num_ubatches=1,
):
    """CPU tensors and an exact first-path HCCL configuration, without a model."""
    config = npu_vllm_config(
        role=role,
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        cudagraph_mode="NONE",
        extra_config=extra_config,
        enable_expert_parallel=False,
        tensor_parallel_size=tensor_parallel_size,
        num_attention_ranks=tensor_parallel_size,
        num_ffn_ranks=tensor_parallel_size,
        enable_dbo=num_ubatches == 2,
        use_ubatching=num_ubatches > 1,
        num_ubatches=num_ubatches,
        ubatch_size=num_ubatches if num_ubatches > 1 else 0,
    )
    config.model_config.dtype = dtype
    config.model_config.is_multimodal_model = False
    config.model_config.multimodal_config = None
    config.model_config.is_encoder_decoder = False
    config.model_config.hf_config.hidden_size = 16
    config.model_config.hf_config.vocab_size = 1024
    config.model_config.hf_config.num_hidden_layers = 2
    config.compilation_config.mode = compilation_mode
    config.compilation_config.cudagraph_mode = cudagraph_mode
    config.compilation_config.use_inductor_graph_partition = False
    config.compilation_config.pass_config = SimpleNamespace(enable_sp=False)
    config.kv_transfer_config = None
    config.lora_config = None
    if dspark:
        native = v41_dspark_config(role=role, tensor_parallel_size=tensor_parallel_size)
        config.speculative_config = native.speculative_config
        config.model_config.model = native.model_config.model
        vars(config.model_config.hf_text_config).update(
            vars(native.model_config.hf_text_config)
        )
    return config


def v41_eager_qualification_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    rank=0,
    tensor_parallel_size=4,
    dspark=False,
    num_ubatches=1,
):
    """Original target config before the FFN worker removes A-owned state."""
    config = hccl_vllm_config(
        dtype=dtype,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        role=role,
        tensor_parallel_size=tensor_parallel_size,
        dspark=dspark,
        num_ubatches=num_ubatches,
    )
    config.parallel_config.rank = rank
    config.model_config.quantization = "deepseek_v4_fp8"
    config.model_config.max_model_len = 1024
    config.model_config.hf_config.num_hidden_layers = 40
    config.quant_config = SimpleNamespace(get_name=lambda: "deepseek_v4_fp8")
    config.load_config = SimpleNamespace(load_format="safetensors")
    config.scheduler_config.max_num_seqs = 1
    config.scheduler_config.max_num_batched_tokens = 1024
    config.scheduler_config.async_scheduling = False
    config.engram_config = SimpleNamespace(
        cpu_offload=True, dp_shared_memory=False, embedding_across_dp=False
    )
    return config


def v41_full_qualification_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    graph=False,
    stages=1,
    dspark=False,
    pd=False,
):
    config = v41_eager_qualification_config(
        dtype=dtype,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        role=role,
        dspark=dspark,
        num_ubatches=stages,
    )
    config.model_config.enforce_eager = not graph
    config.scheduler_config.max_num_seqs = 2
    config.model_config.hf_text_config.engram_layer_ids = [1, 14]
    if dspark:
        config.speculative_config.num_speculative_tokens = 2
    if pd:
        config.kv_transfer_config = pd_kv_transfer_config(tp_size=4)
        config.kv_transfer_config.kv_connector_module_path = (
            "afd_plugin.compat.npu.kv_layout_connector"
        )
    return config


def v41_dummy_qualification_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    num_ubatches=0,
    tensor_parallel_size=1,
    uses_c2=False,
):
    """Small normal-Worker BF16 candidate; metadata only, without device work."""
    config = hccl_vllm_config(
        dtype=dtype,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        role=role,
        num_ubatches=num_ubatches,
        tensor_parallel_size=tensor_parallel_size,
    )
    config.device_config = SimpleNamespace(device_type="npu")
    config.parallel_config.rank = 0
    model = config.model_config
    model.quantization = None
    model.max_model_len = 128
    vars(model.hf_text_config).update(
        hidden_size=512,
        hc_mult=4,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=64,
        kv_lora_rank=512,
        o_lora_rank=64,
        o_groups=1,
        num_hidden_layers=3 if uses_c2 else 2,
        compress_ratios=[0, 0, 2] if uses_c2 else [0, 0],
        sliding_window=128,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        vocab_size=4096,
        intermediate_size=512,
        moe_intermediate_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_hash_layers=0,
        expert_dtype="bf16",
        quantization_config=None,
        engram_layer_ids=[],
    )
    if uses_c2:
        vars(model.hf_text_config).update(
            kv_source_layer_ids=[2],
            index_source_layer_ids=[2],
            candidate_source_layer_id=20,
        )
    config.quant_config = None
    config.load_config = SimpleNamespace(load_format="dummy")
    config.scheduler_config.max_num_seqs = 1
    config.scheduler_config.async_scheduling = False
    config.scheduler_config.enable_chunked_prefill = True
    config.cache_config.block_size = 128
    config.cache_config.cache_dtype = "auto"
    return config


def pd_kv_transfer_config(*, tp_size=1, connector="MooncakeHybridConnector"):
    """Native KVTransferConfig fields only; no transfer engine or remote proof."""
    return SimpleNamespace(
        kv_connector=connector,
        kv_connector_module_path=None,
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "prefill": {"tp_size": tp_size, "dp_size": 1},
            "decode": {"tp_size": tp_size, "dp_size": 1},
        },
    )


def v41_functional_qualification_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    u2=False,
    graph=False,
    dspark=False,
    pd_role=None,
):
    """Bounded non-Engram combination metadata, without worker/device objects."""
    config = v41_dummy_qualification_config(
        dtype=dtype,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        role=role,
        num_ubatches=2 if u2 else 0,
        uses_c2=True,
    )
    config.parallel_config.enable_dbo = False
    config.parallel_config.use_ubatching = u2
    config.model_config.enforce_eager = not graph
    config.model_config.model = "bounded-v41-functional-model"
    config.scheduler_config.max_num_seqs = 2
    if dspark:
        text = config.model_config.hf_text_config
        vars(text).update(
            num_nextn_predict_layers=1,
            dspark_block_size=2,
            dspark_target_layer_ids=[1, 2, 3],
            dspark_n_routed_experts=8,
            dspark_num_experts_per_tok=2,
        )
        draft_text = deepcopy(text)
        draft_text.n_mtp_layers = text.num_nextn_predict_layers
        draft_text.architectures = ["DeepseekV41DSparkModel"]
        config.speculative_config = DSparkSpeculativeConfig(
            method="dspark",
            num_speculative_tokens=2,
            draft_model_config=SimpleNamespace(
                model=config.model_config.model,
                hf_config=draft_text,
                hf_text_config=draft_text,
                architectures=draft_text.architectures,
            ),
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    if pd_role is not None:
        config.kv_transfer_config = pd_kv_transfer_config()
        config.kv_transfer_config.kv_role = pd_role
        config.kv_transfer_config.kv_connector_module_path = (
            "afd_plugin.compat.npu.kv_layout_connector"
        )
    return config


@dataclass
class DSparkSpeculativeConfig:
    method: str
    draft_model_config: SimpleNamespace
    draft_parallel_config: SimpleNamespace
    num_speculative_tokens: int = 5
    rejection_sample_method: str = "standard"
    enable_adaptive_verification: bool = False
    enforce_eager: bool = True
    quantization: str | None = None
    kv_cache_dtype: str | None = "auto"

    def use_dspark(self):
        return self.method == "dspark"


def v41_dspark_config(*, role="attention", tensor_parallel_size=1):
    """Checkpoint-derived metadata; no draft allocation or device execution."""
    config = npu_vllm_config(
        role=role,
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        tensor_parallel_size=tensor_parallel_size,
        num_attention_ranks=tensor_parallel_size,
        num_ffn_ranks=tensor_parallel_size,
    )
    config.model_config.model = "v41-checkpoint"
    text = config.model_config.hf_text_config
    text.num_hidden_layers = 40
    text.n_routed_experts = 384
    text.num_experts_per_tok = 6
    text.num_nextn_predict_layers = 3
    text.dspark_block_size = 5
    text.dspark_target_layer_ids = [37, 38, 39]
    text.dspark_n_routed_experts = 128
    text.dspark_num_experts_per_tok = 3
    draft_text = deepcopy(text)
    draft_text.n_routed_experts = text.dspark_n_routed_experts
    draft_text.num_experts_per_tok = text.dspark_num_experts_per_tok
    draft_text.n_mtp_layers = text.num_nextn_predict_layers
    draft_text.architectures = ["DeepseekV41DSparkModel"]
    draft = SimpleNamespace(
        model=config.model_config.model,
        hf_config=draft_text,
        hf_text_config=draft_text,
        architectures=draft_text.architectures,
    )
    config.speculative_config = DSparkSpeculativeConfig(
        method="dspark",
        draft_model_config=draft,
        draft_parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
        ),
    )
    return config


def v41_dspark_pd_config(*, role="attention", tensor_parallel_size=1):
    """Exact external prefix protocol, with native eager/U1 prerequisites."""
    config = v41_dspark_config(role=role, tensor_parallel_size=tensor_parallel_size)
    config.compilation_config.cudagraph_mode.name = "NONE"
    config.kv_transfer_config = pd_kv_transfer_config(tp_size=tensor_parallel_size)
    config.kv_transfer_config.kv_connector_module_path = (
        "afd_plugin.compat.npu.kv_layout_connector"
    )
    return config


def v41_dspark_pd_graph_config(
    *,
    dtype,
    compilation_mode,
    cudagraph_mode,
    role="attention",
    tensor_parallel_size=1,
    num_ubatches=1,
):
    """Complete original D target configuration; never constructs device owners."""
    config = hccl_vllm_config(
        dtype=dtype,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        role=role,
        tensor_parallel_size=tensor_parallel_size,
        dspark=True,
        num_ubatches=num_ubatches,
    )
    config.model_config.enforce_eager = False
    config.parallel_config.rank = 0
    config.speculative_config.num_speculative_tokens = 2
    config.kv_transfer_config = pd_kv_transfer_config(tp_size=tensor_parallel_size)
    config.kv_transfer_config.kv_connector_module_path = (
        "afd_plugin.compat.npu.kv_layout_connector"
    )
    return config


def native_v41_dbo_config(
    *, role="attention", target_graph=False, dspark=True, boundary=None
):
    """Real native config methods over host-only declared fields.

    Bypass constructors only to isolate configuration validation; this fixture
    does not establish EngineArgs construction or any model/device behavior.
    The actual ParallelConfig properties determine the effective stage count.
    """
    # Deferred imports keep the shared fake module CPU-importable without vLLM.
    import torch
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    source = v41_dspark_pd_graph_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=(
            CUDAGraphMode.FULL_DECODE_ONLY if target_graph else CUDAGraphMode.NONE
        ),
        role=role,
        num_ubatches=2,
    )
    source.model_config.enforce_eager = not target_graph
    source.model_config.is_multimodal_model = True
    source.model_config.is_hybrid = False
    source.model_config.multimodal_config = SimpleNamespace(
        language_model_only=True, mm_encoder_only=False
    )
    source.model_config.hf_text_config.engram_layer_ids = []
    if not dspark:
        source.speculative_config = None
    parallel = object.__new__(ParallelConfig)
    parallel.__dict__.update(
        {
            name: value
            for name, value in vars(source.parallel_config).items()
            if name not in {"num_ubatches", "use_ubatching"}
        }
    )
    config = object.__new__(VllmConfig)
    config.__dict__.update(vars(source))
    # use_v2_model_runner is a real native property, selected by the test env.
    config.__dict__.pop("use_v2_model_runner")
    config.parallel_config = parallel
    if boundary is not None:
        _configure_native_dbo_boundary(config, boundary)
    return config


def _configure_native_dbo_boundary(config, boundary):
    # Deferred with the real-config builder above; no runtime/device objects.
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    if boundary == "no_afd":
        config.additional_config = {}
    elif boundary == "cuda":
        # The test changes the public platform metadata dependency separately.
        pass
    elif boundary in {"cam_p2p", "cam_async"}:
        config.additional_config["afd"]["connector"] = (
            "CAMP2pAFDConnector" if boundary == "cam_p2p" else "CAMAsyncAFDConnector"
        )
    elif boundary == "old_v4":
        config.model_config.hf_config.model_type = "deepseek_v4"
        config.model_config.hf_config.architectures = ["DeepseekV4ForCausalLM"]
    elif boundary in {"u1", "u3", "float_stages"}:
        config.parallel_config.enable_dbo = False
        config.parallel_config.ubatch_size = {"u1": 1, "u3": 3, "float_stages": 2.0}[
            boundary
        ]
    elif boundary == "other_method":
        config.speculative_config.method = "mtp"
    elif boundary == "no_speculation":
        config.speculative_config = None
    elif boundary == "adaptive":
        config.speculative_config.enable_adaptive_verification = True
    elif boundary == "draft_graph":
        config.speculative_config.enforce_eager = False
    elif boundary == "engram_config":
        config.engram_config = SimpleNamespace(cpu_offload=True)
    elif boundary == "engram_layers":
        config.model_config.hf_text_config.engram_layer_ids = [1]
    elif boundary == "vision":
        config.model_config.multimodal_config.language_model_only = False
    elif boundary == "no_mm_config":
        config.model_config.multimodal_config = None
    elif boundary == "compile":
        config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    elif boundary == "graph_partition":
        config.compilation_config.use_inductor_graph_partition = True
    elif boundary == "inconsistent_target_graph":
        config.model_config.enforce_eager = False
    elif boundary == "lora":
        config.lora_config = SimpleNamespace(enabled=True)
    elif boundary == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif boundary == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif boundary == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif boundary == "encoder_decoder":
        config.model_config.is_encoder_decoder = True
    elif boundary == "hybrid":
        config.model_config.is_hybrid = True
    elif boundary == "piecewise":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    elif boundary == "encoder_only":
        config.model_config.multimodal_config.mm_encoder_only = True
    else:
        raise ValueError(f"Unknown native DBO config boundary: {boundary}")


def native_v41_engram_dbo_config(**options):
    from vllm.config import EngramConfig, LoadConfig

    config = native_v41_dbo_config(**options)
    config.model_config.architecture = "DeepseekV41ForCausalLM"
    config.model_config.hf_text_config.engram_layer_ids = [1, 14]
    config.engram_config = EngramConfig(
        cpu_offload=True, embedding_across_dp=False, dp_shared_memory=False
    )
    config.load_config = LoadConfig(load_format="safetensors")
    return config


def native_dbo_config_snapshot(config):
    """Compare declared inputs without native dataclass repr/default fields."""
    snapshot = deepcopy(vars(config))
    snapshot["parallel_config"] = deepcopy(vars(config.parallel_config))
    return snapshot


def async_moe_config(
    *,
    role="attention",
    compute_gate_on_attention=True,
    num_attention_ranks=1,
    num_ffn_ranks=1,
    tensor_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    num_ubatches=2,
    **extra_config,
):
    return npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        use_mla=True,
        async_dp=True,
        compute_gate_on_attention=compute_gate_on_attention,
        num_attention_ranks=num_attention_ranks,
        num_ffn_ranks=num_ffn_ranks,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
        use_ubatching=role == "attention",
        num_ubatches=num_ubatches,
        extra_config={
            "attn_ranks_per_dp": tensor_parallel_size,
            **extra_config,
        },
    )


# DeepEP M2N connector tests: Qwen3-30B-A3B MoE shapes.
M2N_HIDDEN_SIZE = 2048
M2N_NUM_EXPERTS = 128
M2N_TOP_K = 8
M2N_INTERMEDIATE_SIZE = 768
# Fake vLLM limits: capacity defaults to the larger of the two.
M2N_MAX_NUM_BATCHED_TOKENS = 8192
M2N_MAX_CUDAGRAPH_CAPTURE_SIZE = 512
# Fake CUDA-graph capture sizes: 1, 2, 4, then every 8 up to the maximum above.
M2N_CUDAGRAPH_CAPTURE_SIZES = (
    1,
    2,
    4,
    *range(8, M2N_MAX_CUDAGRAPH_CAPTURE_SIZE + 1, 8),
)


@dataclass(frozen=True)
class M2NTopology:
    ffn_size: int
    attn_size: int
    # Attention TP peers (consecutive Attention ranks) share one DP rank's
    # tokens; each dispatches its slice.
    attn_tp_size: int = 1

    @property
    def world_size(self) -> int:
        return self.ffn_size + self.attn_size

    @property
    def attn_dp_size(self) -> int:
        return self.attn_size // self.attn_tp_size

    def role(self, world_rank: int) -> tuple[str, int]:
        if world_rank < self.ffn_size:
            return "ffn", world_rank
        return "attention", world_rank - self.ffn_size


def m2n_vllm_config(
    topology: M2NTopology,
    role: str,
    role_rank: int,
    extra_config: dict[str, Any],
    port: int,
    host: str = "127.0.0.1",
) -> SimpleNamespace:
    """A ``vllm_config`` for one DeepEP M2N rank of ``topology``."""
    # Deferred: this fakes module must import on CPU-only hosts without torch.
    import torch
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    text_config = SimpleNamespace(
        hidden_size=M2N_HIDDEN_SIZE,
        num_hidden_layers=2,
        num_experts=M2N_NUM_EXPERTS,
        num_experts_per_tok=M2N_TOP_K,
        moe_intermediate_size=M2N_INTERMEDIATE_SIZE,
    )
    # vLLM's view: FFN ranks are DP ranks; Attention ranks are DP x TP.
    tp_size = 1 if role == "ffn" else topology.attn_tp_size
    dp_size = topology.ffn_size if role == "ffn" else topology.attn_dp_size
    # The FFN deployment the handshake requires: linear EP over the FFN ranks,
    # with checkpoint weights in the payload's format.
    quant_config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if extra_config.get("payload_format") == "fp8_block128_fp32"
        else None
    )
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "DeepEPM2NAFDConnector",
                "role": role,
                "host": host,
                "port": port,
                "num_attention_ranks": topology.attn_size,
                "num_ffn_ranks": topology.ffn_size,
                "compute_gate_on_attention": True,
                "connector_extra_config": extra_config,
            }
        },
        quant_config=quant_config,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=M2N_MAX_CUDAGRAPH_CAPTURE_SIZE,
            cudagraph_capture_sizes=list(M2N_CUDAGRAPH_CAPTURE_SIZES),
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=role_rank // tp_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_ubatching=False,
            num_ubatches=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=M2N_MAX_NUM_BATCHED_TOKENS
        ),
    )
