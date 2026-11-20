# ruff: noqa: E501
"""Freeze current source for authorized isolated A3 CPU checks; no device use."""

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline-filter", default="ep_")
    args = parser.parse_args()
    root = Path.cwd()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "afd_plugin",
            "tests/unit",
            "tests/conftest.py",
            "tests/__init__.py",
            "tests/e2e/diagnostics",
            "tools/diagnostics",
        ],
        text=True,
    ).splitlines()
    files = {
        name: (root / name).read_bytes()
        for name in names
        if name.endswith(".py") and (root / name).is_file()
    }
    files["cpu-model/config.json"] = (
        root
        / "openspec/changes/afd-hccl-full-parity/evidence/ep-cpu-20261120/checkpoint-config-original.json"
    ).read_bytes()
    if args.baseline:
        with zipfile.ZipFile(
            root / ".tmp/ep-development-20261120/before.zip"
        ) as backup:
            for name in backup.namelist():
                if name.startswith("afd_plugin/"):
                    files[name] = backup.read(name)
    pins = {}
    for package, checkout, revision, relative in (
        (
            "vllm",
            "D:/openSource/vllm",
            "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            "config/parallel.py",
        ),
        (
            "vllm_ascend",
            "D:/openSource/vllm-ascend",
            "7e37e29a7803d6897a7113088e2b771f83205bc0",
            "models/deepseek_v41/model.py",
        ),
    ):
        data = subprocess.check_output(
            ["git", "-C", checkout, "show", revision + ":" + package + "/" + relative]
        )
        pins[package + "/" + relative] = hashlib.sha256(data).hexdigest()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["identity.json"] = json.dumps(
        {
            "manifest": manifest,
            "native_pins": pins,
            "baseline": args.baseline,
            "baseline_filter": args.baseline_filter,
            "hardware_acceptance": False,
        },
        indent=2,
    ).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = buffer.getvalue()
    (out / "source.tar.gz").write_bytes(payload)
    (out / "identity.json").write_bytes(files["identity.json"])
    base_path = (
        root
        / "openspec/changes/afd-hccl-v2-microbatch/evidence/a3-forward-counts-20261109/run_a3_forward_counts.py"
    )
    spec = importlib.util.spec_from_file_location("cpu_base", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    remote = (
        base.ssh(
            'python3 -c \'import tempfile;print(tempfile.mkdtemp(prefix="ep-cpu-",dir="/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"))\''
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), payload)
    (out / "remote.json").write_text(
        json.dumps(
            {"root": remote, "source_sha256": hashlib.sha256(payload).hexdigest()},
            indent=2,
        )
    )
    code = r'''
import pathlib,sys,tarfile,hashlib,json,subprocess,os
root=pathlib.Path(sys.argv[1]); snapshot=root/'snapshot';snapshot.mkdir();results=root/'results';results.mkdir()
with tarfile.open(root/'source.tar.gz') as archive:
 for member in archive:
  name=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not name.is_absolute() and '..' not in name.parts
  archive.extract(member,snapshot,filter='data')
identity=json.loads((snapshot/'identity.json').read_text())
assert all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in identity['manifest'].items())
envscript='source /mnt/workspace/code/.ascend/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null; source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/set_env.sh >/dev/null; export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; /usr/bin/python3 -c "import json,os;print(json.dumps(dict(os.environ)))"'
env=json.loads(subprocess.check_output(['bash','-c',envscript],text=True));env['PYTHONPATH']=str(snapshot)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '');env['VLLM_USE_V2_MODEL_RUNNER']='1';env['AFD_ASYNC_CAM_LAYERED_GMM']='0'
child=r"""
import pathlib,json,torch,torch_npu,pytest,hashlib,importlib.util,importlib.metadata
from unittest.mock import patch
identity=json.loads(pathlib.Path('identity.json').read_text());before=torch.npu.is_initialized();assert not before
matches={}
for name, expected in identity['native_pins'].items():
 package,relative=name.split('/',1);source=pathlib.Path(importlib.util.find_spec(package).origin).parent/relative
 matches[name]=hashlib.sha256(source.read_bytes()).hexdigest()==expected
assert all(matches.values()), matches
packages={name:importlib.metadata.version(name) for name in ('torch','torch-npu','vllm','vllm-ascend')}
targets=['tests/unit/distributed/test_hccl.py','tests/unit/distributed/test_hccl_plan.py','tests/unit/connectors/npu/test_hccl_config.py','tests/unit/connectors/npu/test_hccl_staged.py','tests/unit/connectors/npu/test_hccl_staged_transport.py','tests/unit/connectors/npu/test_p2p_hccl.py','tests/unit/v1/worker/npu/test_hccl_v41_ffn_runner.py','tests/unit/v1/worker/npu/test_hccl_v41_execution.py','tests/unit/v1/worker/npu/test_ffn_model_runner.py','tests/unit/compat/npu/test_feature_validation.py','tests/unit/compat/npu/test_runtime_config.py','tests/unit/entrypoints/test_npu_ffn.py','tests/e2e/diagnostics/test_dsv41_full_checkpoint.py']
with patch.object(torch.npu,'_lazy_init',side_effect=RuntimeError('CPU test forbids NPU initialization')):
 code=int(pytest.main(targets+(['-k',identity['baseline_filter']] if identity['baseline'] else [])+['-q','-p','no:cacheprovider','--tb=short','--junitxml=../results/junit.xml']))
after=torch.npu.is_initialized();pathlib.Path('../results/state.json').write_text(json.dumps({'exit_code':code,'npu_initialized_before':before,'npu_initialized_after':after,'native_pins_match':matches,'packages':packages},indent=2));raise SystemExit(code or int(after))
"""
run=subprocess.run(['/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python','-B','-c',child],cwd=snapshot,env=env,capture_output=True,timeout=240)
(results/'pytest.log').write_bytes(run.stdout+run.stderr)
if not identity['baseline']:
 config=subprocess.run(['/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python','-B','tools/diagnostics/check_dsv41_full_checkpoint_config.py','--expert-parallel','--model',str(snapshot/'cpu-model'),'--output',str(results/'native-config.json')],cwd=snapshot,env=env,capture_output=True,timeout=180)
 (results/'native-config.log').write_bytes(config.stdout+config.stderr)
 print('native_config_exit',config.returncode)
 if config.returncode: print((config.stdout+config.stderr).decode(errors='replace')[-7000:])
 run.returncode = run.returncode or config.returncode
unchanged=all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in identity['manifest'].items())
(results/'summary.json').write_text(json.dumps({'exit_code':run.returncode,'source_unchanged':unchanged,'hardware_acceptance':False},indent=2))
print((run.stdout+run.stderr).decode(errors='replace')[-12000:]);print('source_unchanged',unchanged)
raise SystemExit(run.returncode or int(not unchanged))
'''
    (out / "remote-runner.py").write_text(code)
    run = subprocess.run(
        [
            "ssh",
            "-F",
            base.SSH_CONFIG,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            base.SSH_HOST,
            "python3 -B - " + shlex.quote(remote),
        ],
        input=code.encode(),
        capture_output=True,
        timeout=300,
    )
    (out / "remote.log").write_bytes(run.stdout + run.stderr)
    print(run.stdout.decode(errors="replace"))
    print(run.stderr.decode(errors="replace"))
    (out / "results.tar.gz").write_bytes(
        base.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    )
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
