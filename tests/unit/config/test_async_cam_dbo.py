# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import configure_async_cam_dbo


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("native_flag", [False, True])
def test_native_ubatching_uses_dbo_at_every_dp_size(dp_size, native_flag):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            use_ubatching=native_flag,
            enable_dbo=False,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=8,
        ),
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async_dp": True,
            }
        },
    )
    configure_async_cam_dbo(config)
    assert config.parallel_config.enable_dbo is native_flag
    configure_async_cam_dbo(config)
    assert config.parallel_config.enable_dbo is native_flag


@pytest.mark.parametrize(
    "role,connector",
    [("ffn", "CAMAsyncAFDConnector"), ("attention", "CAMP2pAFDConnector")],
)
def test_async_normalization_does_not_enable_ffn_or_synchronous_dbo(role, connector):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True, enable_dbo=False),
        additional_config={
            "afd": {
                "connector": connector,
                "role": role,
            }
        },
    )
    configure_async_cam_dbo(config)
    assert not config.parallel_config.enable_dbo


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize(
    "threshold", ["dbo_decode_token_threshold", "dbo_prefill_token_threshold"]
)
@pytest.mark.parametrize("value", [0, 1])
def test_native_ubatching_revalidates_thresholds(dp_size, threshold, value):
    parallel = SimpleNamespace(
        data_parallel_size=dp_size,
        use_ubatching=True,
        enable_dbo=False,
        num_ubatches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=8,
    )
    setattr(parallel, threshold, value)
    config = SimpleNamespace(
        parallel_config=parallel,
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async_dp": True,
            }
        },
    )
    with pytest.raises(ValueError, match="at least the number of microbatches"):
        configure_async_cam_dbo(config)
