# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from afd_plugin.config import HCCL_P2P_CONNECTOR, AFDConfig


def camp2p_config(*, role: str):
    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )


def hccl_config(*, role="attention", attention_size=1, ffn_size=1):
    return AFDConfig(
        connector=HCCL_P2P_CONNECTOR,
        role=role,
        num_attention_ranks=attention_size,
        num_ffn_ranks=ffn_size,
    )
