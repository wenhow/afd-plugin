import gzip
import io
import json
import struct
from pathlib import Path

import pytest

from tools import collect_dsv41_tensor_metadata as collector


def write_model(tmp_path, *, architecture="DeepseekV41ForCausalLM", tensors=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": "deepseek_v41"})
    )
    tensors = tensors or {
        "model.layers.1.engram.weight": {
            "dtype": "F4",
            "shape": [4],
            "data_offsets": [0, 2],
        },
        "model.layers.0.ffn.weight": {
            "dtype": "U8",
            "shape": [2],
            "data_offsets": [2, 4],
        },
    }
    raw = json.dumps(tensors).encode()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
    index = {
        "metadata": {"total_size": 4},
        "weight_map": dict.fromkeys(tensors, shard.name),
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    return model, shard


def test_collect_preserves_names_offsets_and_subbyte_dtype(tmp_path):
    model, _ = write_model(tmp_path)
    result = collector.collect_metadata(model)
    assert result["complete"] is True
    assert result["total_tensor_bytes"] == 4
    assert result["tensor_count"] == 2
    tensor = result["shards"][0]["tensors"]["model.layers.1.engram.weight"]
    assert tensor == {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}


def test_collection_never_reads_tensor_payload(tmp_path, monkeypatch):
    model, shard = write_model(tmp_path)
    content = shard.read_bytes()
    boundary = collector.HEADER_PREFIX_BYTES + struct.unpack("<Q", content[:8])[0]
    original_open = Path.open

    class HeaderOnlyReader(io.BytesIO):
        def read(self, size=-1):
            assert size >= 0
            assert self.tell() + size <= boundary
            return super().read(size)

    def open_header(path, *args, **kwargs):
        if path == shard:
            return HeaderOnlyReader(content)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_header)
    assert collector.collect_metadata(model)["complete"] is True


def test_wrong_model_family_is_rejected(tmp_path):
    model, _ = write_model(tmp_path, architecture="DeepseekV4ForCausalLM")
    with pytest.raises(ValueError, match="DeepseekV41ForCausalLM"):
        collector.collect_metadata(model)


@pytest.mark.parametrize("corruption", ["missing", "truncated", "oversized", "offsets"])
def test_bad_shards_produce_incomplete_report(tmp_path, corruption):
    model, shard = write_model(tmp_path)
    if corruption == "missing":
        shard.unlink()
    elif corruption == "truncated":
        shard.write_bytes(b"short")
    elif corruption == "oversized":
        shard.write_bytes(struct.pack("<Q", collector.MAX_HEADER_BYTES + 1))
    else:
        raw = json.dumps(
            {"weight": {"dtype": "U8", "shape": [5], "data_offsets": [0, 5]}}
        ).encode()
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
    result = collector.collect_metadata(model)
    assert result["complete"] is False
    assert result["errors"]


def test_index_tensor_owner_mismatch_is_reported(tmp_path):
    model, _ = write_model(tmp_path)
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["missing.weight"] = next(iter(index["weight_map"].values()))
    index_path.write_text(json.dumps(index))
    assert collector.collect_metadata(model)["complete"] is False


def test_output_cannot_overwrite_checkpoint(tmp_path):
    model, _ = write_model(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        collector.write_report({}, model / "config.json", model)


def test_gzip_output_round_trip(tmp_path):
    model, _ = write_model(tmp_path)
    report = collector.collect_metadata(model)
    output = tmp_path / "tensors.json.gz"
    collector.write_report(report, output, model)
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert json.load(stream) == report


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/x.safetensors"])
def test_index_cannot_escape_model_directory(tmp_path, filename):
    model, _ = write_model(tmp_path)
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = dict.fromkeys(index["weight_map"], filename)
    index_path.write_text(json.dumps(index))
    assert collector.collect_metadata(model)["complete"] is False


def test_duplicate_tensor_names_are_rejected(tmp_path):
    model, shard = write_model(tmp_path)
    raw = b'{"weight": {}, "weight": {}}'
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert collector.collect_metadata(model)["complete"] is False


def test_overlapping_tensor_storage_is_rejected(tmp_path):
    model, shard = write_model(tmp_path)
    tensors = {
        "left": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
        "right": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]},
    }
    raw = json.dumps(tensors).encode()
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
    assert collector.collect_metadata(model)["complete"] is False


def test_index_total_size_mismatch_is_rejected(tmp_path):
    model, _ = write_model(tmp_path)
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] += 1
    index_path.write_text(json.dumps(index))
    assert collector.collect_metadata(model)["complete"] is False


def test_existing_output_is_preserved(tmp_path):
    model, _ = write_model(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        collector.write_report({}, output, model)
    assert output.read_text() == "preserve"
