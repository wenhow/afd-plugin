# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import TestCase

from afd_plugin.v1.worker.npu.window_dspark import configure_window_dspark_draft


class WindowDSparkDraftConfigTest(TestCase):
    def test_embedded_checkpoint_uses_its_draft_router(self):
        for target_experts, draft_experts in ((384, 128), (72, 128), (384, 64)):
            with self.subTest(target_experts=target_experts, draft_experts=draft_experts):
                target = SimpleNamespace(
                    model="view",
                    hf_text_config=SimpleNamespace(
                        n_routed_experts=target_experts, num_experts_per_tok=6,
                        dspark_n_routed_experts=target_experts, dspark_num_experts_per_tok=6,
                        num_hidden_layers=20,
                    ),
                )
                draft = SimpleNamespace(
                    model="view", hf_config=SimpleNamespace(),
                    hf_text_config=SimpleNamespace(
                        n_routed_experts=384, num_experts_per_tok=6, num_hidden_layers=40,
                        dspark_n_routed_experts=draft_experts, dspark_num_experts_per_tok=3,
                    ),
                )
                configure_window_dspark_draft(target, draft)
                self.assertEqual(draft.hf_text_config.n_routed_experts, draft_experts)
                self.assertEqual(draft.hf_text_config.num_experts_per_tok, 3)
                self.assertEqual(draft.hf_text_config.num_hidden_layers, 20)
                self.assertEqual(target.hf_text_config.n_routed_experts, target_experts)
                self.assertEqual(target.hf_text_config.num_experts_per_tok, 6)
                self.assertEqual(draft.hf_config.architectures, ["AFDDeepseekV41DSparkModel"])

    def test_standalone_checkpoint_uses_its_own_config(self):
        for overrides in ({}, {"dspark_n_routed_experts": 64, "dspark_num_experts_per_tok": 2}):
            with self.subTest(overrides=overrides):
                draft = SimpleNamespace(
                    model="draft", hf_config=SimpleNamespace(),
                    hf_text_config=SimpleNamespace(
                        n_routed_experts=96, num_experts_per_tok=4, num_hidden_layers=32,
                        **overrides,
                    ),
                )
                configure_window_dspark_draft(SimpleNamespace(model="target"), draft)
                self.assertEqual(draft.hf_text_config.n_routed_experts, 64 if overrides else 96)
                self.assertEqual(draft.hf_text_config.num_experts_per_tok, 2 if overrides else 4)
                self.assertEqual(draft.hf_text_config.num_hidden_layers, 32)
