"""Read-only follow-up to the preserved seven-idle-board first sample."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path

OBSERVED_PID = 4_115_288
BASE_HARNESS = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-forward-counts-20261109/run_a3_forward_counts.py"
)


def main():
    attempt = Path(__file__).resolve().parent
    root = attempt.parents[5]
    spec = importlib.util.spec_from_file_location(
        "existing_a3_harness", root / BASE_HARNESS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owned = json.loads((attempt / "results/owned-processes.json").read_text())[
        "processes"
    ]
    identity = json.loads((attempt / "identity.json").read_text())
    snapshot = identity["remote"] + "/snapshot"
    script = """import importlib.util,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
spec=importlib.util.spec_from_file_location('existing_a3_harness', REMOTE_HARNESS)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
env=module.prepare_env(Path(SNAPSHOT))
pids = PID_LIST
processes = {}
for pid in pids:
    path = Path('/proc') / str(pid) / 'stat'
    try:
        fields = path.read_text().rsplit(') ', 1)[1].split()
        processes[str(pid)] = {'exists': True, 'state': fields[0],
            'ppid': int(fields[1]), 'pgid': int(fields[2]),
            'start_ticks': fields[19]}
    except FileNotFoundError:
        processes[str(pid)] = {'exists': False}
result = subprocess.run(['npu-smi','info'],env=env,capture_output=True,timeout=30)
print(json.dumps({'scope':'read-only postcheck; no process signals',
    'collected_utc':datetime.now(timezone.utc).isoformat(),
    'observed_first_sample_pid':OBSERVED,
    'observed_pid_belongs_to_bank_probe':MEMBER,
    'processes':processes,'npu_smi_returncode':result.returncode,
    'idle_boards':result.stdout.count(b'No running processes found in NPU'),
    'npu_smi':(result.stdout+result.stderr).decode(errors='replace')},indent=2))
"""
    script = script.replace("PID_LIST", repr([OBSERVED_PID, *map(int, owned)]))
    script = script.replace("REMOTE_HARNESS", repr(snapshot + "/" + BASE_HARNESS))
    script = script.replace("SNAPSHOT", repr(snapshot))
    script = script.replace("OBSERVED", str(OBSERVED_PID))
    script = script.replace("MEMBER", repr(str(OBSERVED_PID) in owned))
    output = module.ssh("python3 -B -c " + shlex.quote(script))
    destination = attempt / "postcheck-002.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(output)
    result = json.loads(output)
    print(
        json.dumps({key: result[key] for key in result if key != "npu_smi"}, indent=2)
    )


if __name__ == "__main__":
    main()
