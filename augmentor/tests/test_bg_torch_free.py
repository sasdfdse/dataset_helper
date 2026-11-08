"""GUI/CLI/워크플로 경계는 torch 계열을 import하지 않는다 (무거운 모델 코드는 background_swap.models만).

같은 pytest 프로세스에는 다른 테스트가 이미 torch를 올려 두었을 수 있으므로 새 하위 프로세스에서
확인한다. 현재 인터프리터를 그대로 쓰므로 torch가 설치된 venv에서도 의미 있는 검사다.
"""
import subprocess
import sys
from pathlib import Path

AUGMENTOR = Path(__file__).resolve().parent.parent

_CODE = (
    "import sys, gui.app, gui.background_tab, background_swap.cli, "
    "background_swap.pipeline.workflow; "
    "bad=[m for m in ('torch','transformers','diffusers','bitsandbytes') if m in sys.modules]; "
    "assert not bad, bad"
)


def test_gui_cli_workflow_import_without_torch():
    r = subprocess.run([sys.executable, "-c", _CODE], cwd=AUGMENTOR, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
