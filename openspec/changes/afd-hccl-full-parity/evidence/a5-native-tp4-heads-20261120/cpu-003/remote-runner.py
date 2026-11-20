
import pathlib,sys,tarfile,hashlib,json,subprocess,os
root=pathlib.Path(sys.argv[1]); snapshot=root/'snapshot'; snapshot.mkdir(); results=root/'results'; results.mkdir()
with tarfile.open(root/'source.tar.gz') as archive:
 for member in archive:
  name=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not name.is_absolute() and '..' not in name.parts
  archive.extract(member,snapshot,filter='data')
identity=json.loads((snapshot/'identity.json').read_text())
assert all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in identity['manifest'].items())
envscript='source /mnt/workspace/code/.ascend/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null; source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/set_env.sh >/dev/null; export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; /usr/bin/python3 -c "import json,os;print(json.dumps(dict(os.environ)))"'
env=json.loads(subprocess.check_output(['bash','-c',envscript],text=True)); env['PYTHONPATH']=str(snapshot)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '');env['VLLM_USE_V2_MODEL_RUNNER']='1';env['AFD_ASYNC_CAM_LAYERED_GMM']='0'
child=r"""
import hashlib,pathlib,json,importlib.util,torch,torch_npu,pytest
from unittest.mock import patch
source=pathlib.Path(importlib.util.find_spec('vllm_ascend').origin).parent
identity=json.loads(pathlib.Path('identity.json').read_text())
matches={n:hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in identity['pins'].items()}
assert all(matches.values()), matches
before=torch.npu.is_initialized();assert not before
with patch.object(torch.npu,'_lazy_init',side_effect=RuntimeError('CPU test forbids NPU initialization')):
 code=int(pytest.main(['tests/unit/v1/worker/npu/test_dsv41_native_worker.py','tests/unit/model_executor/models/npu/test_dsv41_packed_cache.py','-q','-p','no:cacheprovider','--tb=short','--junitxml=../results/junit.xml']))
after=torch.npu.is_initialized()
pathlib.Path('../results/state.json').write_text(json.dumps({'exit_code':code,'npu_initialized_before':before,'npu_initialized_after':after,'native_pins_match':matches},indent=2))
raise SystemExit(code or int(after))
"""
run=subprocess.run(['/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python','-B','-c',child],cwd=snapshot,env=env,capture_output=True,timeout=180)
(results/'pytest.log').write_bytes(run.stdout+run.stderr)
unchanged=all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in identity['manifest'].items())
(results/'summary.json').write_text(json.dumps({'exit_code':run.returncode,'source_unchanged':unchanged,'hardware_acceptance':False},indent=2))
print((run.stdout+run.stderr).decode(errors='replace')[-5000:]); print('source_unchanged',unchanged)
raise SystemExit(run.returncode or int(not unchanged))
