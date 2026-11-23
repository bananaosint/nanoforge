"""Exercise installer routing without downloading large wheels or launching servers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

START = Path(__file__).resolve().parents[2] / 'start.sh'


def fixture_project(tmp_path, platform='Linux', torch_ready=False, profile='cpu', fail_pip=False):
    root = tmp_path / 'project with spaces'
    (root / '.venv/bin').mkdir(parents=True)
    (root / 'backend').mkdir()
    (root / 'frontend').mkdir()
    (root / 'fake-bin').mkdir()
    (root / 'backend/requirements.txt').write_text('torch>=2.4,<3\n')
    (root / 'start.sh').write_text(START.read_text())
    fake = f'''#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['INSTALL_LOG'], 'a') as log:
    log.write(json.dumps(dict(name=name, args=sys.argv[1:], temp=os.environ.get('TMPDIR'), cache=os.environ.get('PIP_NO_CACHE_DIR'))) + '\\n')
if name == 'uname':
    print(os.environ['FAKE_PLATFORM'])
if name == 'python' and sys.argv[1:2] == ['-c']:
    sys.exit(0 if os.environ['FAKE_TORCH_READY'] == '1' else 1)
if name == 'python' and 'pip' in sys.argv and os.environ['FAKE_FAIL_PIP'] == '1':
    sys.exit(1)
'''
    for path in (root / '.venv/bin/python', root / 'fake-bin/npm', root / 'fake-bin/uname'):
        path.write_text(fake)
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(root / 'fake-bin') + os.pathsep + os.environ['PATH'],
           'NANOFORGE_SETUP_ONLY': '1', 'NANOFORGE_TORCH': profile,
           'INSTALL_LOG': str(root / 'install.jsonl'), 'FAKE_PLATFORM': platform,
           'FAKE_TORCH_READY': '1' if torch_ready else '0', 'FAKE_FAIL_PIP': '1' if fail_pip else '0',
           'TMPDIR': str(tmp_path / 'simulated-small-tmpfs')}
    return root, env


def run(root, env):
    result = subprocess.run(['bash', str(root / 'start.sh')], env=env, text=True, capture_output=True)
    log = root / 'install.jsonl'
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


@pytest.mark.parametrize('platform,ready,profile,cpu_wheel', [
    ('Linux', False, 'cpu', True),
    ('Darwin', False, 'cpu', False),
    ('Linux', False, 'cuda', False),
    ('Linux', True, 'cpu', False),
])
def test_installer_routing_and_disk_temp(tmp_path, platform, ready, profile, cpu_wheel):
    root, env = fixture_project(tmp_path, platform, ready, profile)
    result, calls = run(root, env)
    assert result.returncode == 0, result.stderr
    pip = [c for c in calls if c['name'] == 'python' and 'pip' in c['args']]
    cpu = [c for c in pip if '--index-url' in c['args']]
    assert bool(cpu) == cpu_wheel
    if cpu_wheel:
        assert cpu[0]['args'][-1] == 'https://download.pytorch.org/whl/cpu'
    assert len(pip) == (2 if cpu_wheel else 1)
    assert all(c['temp'] == str(root / '.venv/install-tmp') for c in calls)
    assert all(c['cache'] == '1' for c in calls)
    assert all('--no-cache-dir' in c['args'] for c in pip)
    assert any(c['name'] == 'npm' and c['args'] == ['ci'] for c in calls)
    assert (root / '.venv/.nanoforge-ready').read_bytes() == (root / 'backend/requirements.txt').read_bytes()
    assert 'NanoForge dependencies are ready.' in result.stdout


def test_failed_download_does_not_mark_environment_ready(tmp_path):
    root, env = fixture_project(tmp_path, fail_pip=True)
    result, calls = run(root, env)
    assert result.returncode != 0
    assert not (root / '.venv/.nanoforge-ready').exists()
    assert not any(c['name'] == 'npm' for c in calls)


def test_ready_environment_skips_downloads(tmp_path):
    root, env = fixture_project(tmp_path)
    (root / '.venv/.nanoforge-ready').write_bytes((root / 'backend/requirements.txt').read_bytes())
    (root / 'frontend/node_modules').mkdir()
    result, calls = run(root, env)
    assert result.returncode == 0
    assert not calls


def test_invalid_profile_fails_before_installing(tmp_path):
    root, env = fixture_project(tmp_path, profile='unknown')
    result, calls = run(root, env)
    assert result.returncode != 0
    assert not calls
    assert 'must be cpu, cuda, or xpu' in result.stderr


def test_xpu_profile_replaces_cpu_even_with_ready_marker(tmp_path):
    root, env = fixture_project(tmp_path, profile='xpu', torch_ready=False)
    (root / '.venv/.nanoforge-ready').write_bytes((root / 'backend/requirements.txt').read_bytes())
    result, calls = run(root, env)
    assert result.returncode == 0, result.stderr
    pip = [c for c in calls if c['name']=='python' and 'pip' in c['args']]
    assert len(pip)==1
    assert pip[0]['args'][-1]=='https://download.pytorch.org/whl/xpu'
    assert '--force-reinstall' in pip[0]['args']
    assert any(c['args']==['-m','backend.check_device','--device','xpu'] for c in calls)


def test_xpu_profile_keeps_existing_xpu_wheel(tmp_path):
    root, env = fixture_project(tmp_path, profile='xpu', torch_ready=True)
    (root / '.venv/.nanoforge-ready').write_bytes((root / 'backend/requirements.txt').read_bytes())
    result, calls = run(root, env)
    assert result.returncode == 0
    assert not any('pip' in c['args'] for c in calls)
    assert any(c['args']==['-m','backend.check_device','--device','xpu'] for c in calls)
