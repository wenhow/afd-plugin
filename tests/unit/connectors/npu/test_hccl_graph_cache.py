# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU ownership contracts; no simulated graph/device acceptance."""

import gc
import weakref
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.distributed.hccl_graph import HcclGraphStage
from tests.unit.fakes.graph_cache import DeviceCompletion, GraphResources, ReplayGraph
from tests.unit.fakes.hccl_graph_control import make_graph_key


@pytest.fixture
def key():
    return make_graph_key(rows=(4, 4))


@pytest.fixture
def cache():
    return HcclGraphCache(DeviceCompletion().synchronize)


def test_pending_capture_is_a_miss_until_device_completion(cache, key):
    graph, resources = ReplayGraph(), GraphResources(3)
    cache.reserve_capture(key, 0, graph, resources)
    assert cache.contains(key)
    assert cache.find(key, 0) is None
    cache.publish_capture(key, 0)
    entry = cache.find(key, 0)
    assert entry is not None
    assert (entry.key, entry.generation, entry.resources.value) == (key, 0, 3)
    entry.graph.replay()


def test_cache_is_empty_only_before_reservation_or_after_release(cache, key):
    assert cache.is_empty
    cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    assert not cache.is_empty
    cache.publish_capture(key, 0)
    assert not cache.is_empty
    cache.fail()
    assert not cache.is_empty
    cache.close()
    assert cache.is_empty


def test_storage_generation_change_is_a_miss(cache, key):
    cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    cache.publish_capture(key, 0)
    assert cache.find(key, 1) is None
    assert cache.contains(key)
    assert cache.find(key, 0) is not None


def test_same_token_bucket_with_different_request_geometry_is_a_miss(cache, key):
    cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    cache.publish_capture(key, 0)
    different = replace(
        key, stage_layouts=(key.stage_layouts[0], HcclGraphStage(3, 8, 4, 8))
    )
    assert cache.find(different, 0) is None


@pytest.mark.parametrize("published", [False, True])
def test_duplicate_capture_requires_a_new_session(cache, key, published):
    cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    if published:
        cache.publish_capture(key, 0)
    with pytest.raises(ValueError, match="already reserved"):
        cache.reserve_capture(key, 1, ReplayGraph(), GraphResources())


@pytest.mark.parametrize("generation", [-1, True, 1.0])
def test_invalid_storage_generation_is_rejected(cache, key, generation):
    with pytest.raises(ValueError, match="generation"):
        cache.reserve_capture(key, generation, ReplayGraph(), GraphResources())


def test_wrong_publication_generation_keeps_entry_unavailable(cache, key):
    cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    with pytest.raises(ValueError, match="reserved key and generation"):
        cache.publish_capture(key, 1)
    assert cache.find(key, 0) is None


def test_failed_device_completion_never_advertises_a_capture(key):
    completion = DeviceCompletion()
    cache = HcclGraphCache(completion.synchronize)
    resources = GraphResources()
    lifetime = weakref.ref(resources)
    cache.reserve_capture(key, 0, ReplayGraph(), resources)
    del resources
    completion.failed = True
    with pytest.raises(RuntimeError, match="device completion failed"):
        cache.publish_capture(key, 0)
    with pytest.raises(RuntimeError, match="not active"):
        cache.find(key, 0)
    gc.collect()
    assert lifetime() is not None
    completion.failed = False
    cache.close()
    gc.collect()
    assert lifetime() is None


@pytest.mark.parametrize("published", [False, True])
def test_close_releases_capture_dependencies_after_reset(cache, key, published):
    graph, resources = ReplayGraph(), GraphResources()
    lifetime = weakref.ref(resources)
    cache.reserve_capture(key, 0, graph, resources)
    if published:
        cache.publish_capture(key, 0)
    del resources
    gc.collect()
    assert lifetime() is not None
    cache.close()
    gc.collect()
    assert lifetime() is None
    with pytest.raises(RuntimeError, match="graph was reset"):
        graph.replay()
    cache.close()


def test_reset_failure_retains_dependencies_for_supervision(cache, key):
    resources = GraphResources()
    lifetime = weakref.ref(resources)
    cache.reserve_capture(key, 0, ReplayGraph(reset_error=True), resources)
    del resources
    with pytest.raises(RuntimeError, match="graph reset failed"):
        cache.close()
    gc.collect()
    assert lifetime() is not None
    assert cache.closed is False
    with pytest.raises(RuntimeError, match="not active"):
        cache.find(key, 0)


def test_failed_cache_cannot_be_reused(cache, key):
    cache.fail()
    with pytest.raises(RuntimeError, match="not active"):
        cache.reserve_capture(key, 0, ReplayGraph(), GraphResources())
    cache.close()
