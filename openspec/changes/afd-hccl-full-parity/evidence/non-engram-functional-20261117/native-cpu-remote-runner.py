import pathlib,tarfile,json,hashlib,subprocess,sys,os
root=pathlib.Path(sys.argv[1]);snapshot=root/'snapshot';snapshot.mkdir();results=root/'results';results.mkdir()
with tarfile.open(root/'source.tar.gz') as t:
 for m in t:
  p=pathlib.PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  t.extract(m,snapshot,filter='data')
manifest=json.loads((snapshot/'manifest.json').read_text())
assert all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
envscript='source /mnt/workspace/code/.ascend/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null; source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/set_env.sh >/dev/null; export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; /usr/bin/python3 -c "import json,os;print(json.dumps(dict(os.environ)))"'
env=json.loads(subprocess.check_output(['bash','-c',envscript],text=True));env['PYTHONPATH']=str(snapshot)+':'+str(snapshot/'tests/npu')+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
env['VLLM_USE_V2_MODEL_RUNNER']='1';env['AFD_ASYNC_CAM_LAYERED_GMM']='0'
child="""import torch,torch_npu,pytest,json,pathlib
before=torch.npu.is_initialized()
code=int(pytest.main(['tests/unit/compat/npu/test_feature_validation.py','tests/unit/compat/npu/test_runtime_config.py','tests/unit/v1/worker/npu/test_ffn_worker.py','tests/unit/compat/patches/test_engine_core.py','tests/unit/compat/npu/test_profiler.py','tests/unit/compat/npu/test_profiler_api.py','tests/unit/compat/patches/test_signatures.py','-q','-o','addopts=','-p','no:cacheprovider','--tb=short','--junitxml=../results/junit.xml']))
pathlib.Path('../results/native-state.json').write_text(json.dumps({'before':before,'after':torch.npu.is_initialized(),'returncode':code}))
raise SystemExit(code or int(before or torch.npu.is_initialized()))
"""
run=subprocess.run(['/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python','-B','-c',child],cwd=snapshot,env=env,capture_output=True,timeout=180)
(results/'pytest.log').write_bytes(run.stdout+run.stderr)
import xml.etree.ElementTree as ET
counts={'tests':0,'failures':0,'errors':0,'skipped':0}
if (results/'junit.xml').is_file():
 for c in ET.parse(results/'junit.xml').iter('testcase'):
  counts['tests']+=1
  for key,tag in [('failures','failure'),('errors','error'),('skipped','skipped')]:counts[key]+=c.find(tag) is not None
summary={'returncode':run.returncode,'junit':counts,'source_unchanged':all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in manifest.items())}
if (results/'native-state.json').is_file():summary['native_state']=json.loads((results/'native-state.json').read_text())
(results/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));print((run.stdout+run.stderr).decode(errors='replace')[-6000:])
raise SystemExit(run.returncode)
