import dataclasses
import io
import json
import sys

from background_swap.cli import main
from background_swap.domain import GpuOutOfMemory
from background_swap.settings import BackgroundSwapSettings, save_settings
from tests.bg_fakes import FakeFactory, FakeInpainter, make_settings

_settings = make_settings


def _events(buf):
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def test_run_success_emits_progress_and_done(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(), buf) == 0
    ev = _events(buf)
    assert ev[-1] == {"event": "done"} and any(e.get("stage") == "inpaint" for e in ev)


def test_class_names_warning_when_missing(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(), buf) == 0
    assert {"event": "warning",
            "message": "클래스 이름 파일이 없어 새 객체 검사를 건너뜁니다."} in _events(buf)


def test_invalid_settings_exit_2(tmp_path):
    p = tmp_path / "s.json"
    save_settings(BackgroundSwapSettings(), p)
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(), buf) == 2
    assert _events(buf)[-1]["event"] == "error"
    assert _events(buf)[-1]["code"] == 2


def test_non_train_exit_3(tmp_path):
    s = _settings(tmp_path / "val", n_images=1)
    p = tmp_path / "s.json"
    save_settings(s, p)
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(), buf) == 3
    assert _events(buf)[-1]["code"] == 3


def test_settings_mismatch_exit_4(tmp_path):
    s = _settings(tmp_path, n_images=1)
    save_settings(s, s.output_dir / "run_settings.json")
    other = tmp_path / "other.json"
    save_settings(dataclasses.replace(s, seed=9), other)
    buf = io.StringIO()
    assert main(["run", "--settings", str(other)], lambda s, c: FakeFactory.default(), buf) == 4
    assert "seed" in _events(buf)[-1]["message"]


def test_oom_exit_6(tmp_path):
    s = _settings(tmp_path, n_images=1)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    f = FakeFactory.default(inpainter=FakeInpainter(raise_exc=GpuOutOfMemory()))
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: f, buf) == 6
    assert "VRAM" in _events(buf)[-1]["message"]


def test_other_exception_exit_1(tmp_path, capsys):
    s = _settings(tmp_path, n_images=1)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)

    def boom(s, c):
        raise RuntimeError("kaput")

    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], boom, buf) == 1
    last = _events(buf)[-1]
    assert last["code"] == 1 and "RuntimeError" in last["message"] and "kaput" in last["message"]
    assert "Traceback" in capsys.readouterr().err


def test_missing_heavy_deps_exit_5(tmp_path, monkeypatch):
    s = _settings(tmp_path, n_images=1)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    monkeypatch.setitem(sys.modules, "background_swap.models", None)    # import 시 ImportError
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], None, buf) == 5
    assert "requirements-bg.txt" in _events(buf)[-1]["message"]


def test_preflight_failure_exit_5(tmp_path, monkeypatch):
    import types
    s = _settings(tmp_path, n_images=1)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    fake = types.ModuleType("background_swap.models")
    fake.HFModelFactory = lambda s, c: None
    fake.preflight = lambda s: ["모델이 없습니다: x"]
    monkeypatch.setitem(sys.modules, "background_swap.models", fake)
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], None, buf) == 5
    assert "모델이 없습니다: x" in _events(buf)[-1]["message"]


def test_preview_and_export(tmp_path):
    s = _settings(tmp_path, n_images=2, use_vlm=False)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    buf = io.StringIO()
    assert main(["preview", "--settings", str(p), "--n", "1"],
                lambda s, c: FakeFactory.default(), buf) == 0
    assert _events(buf)[-1]["preview_dir"].endswith("_preview")
    main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(), io.StringIO())
    dst = tmp_path / "dst" / "train"
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    buf = io.StringIO()
    assert main(["export", "--settings", str(p), "--images-dir", str(dst / "images"),
                 "--labels-dir", str(dst / "labels")],
                lambda s, c: FakeFactory.default(), buf) == 0
    assert _events(buf)[-1]["copied"] == 6


def test_export_to_non_train_exit_3(tmp_path):
    s = _settings(tmp_path, n_images=1)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    dst = tmp_path / "dst" / "val"
    buf = io.StringIO()
    assert main(["export", "--settings", str(p), "--images-dir", str(dst / "images"),
                 "--labels-dir", str(dst / "labels")], None, buf) == 3
    assert _events(buf)[-1]["code"] == 3


def test_prints_during_model_import_go_to_stderr(tmp_path, monkeypatch, capsys):
    import importlib.abc
    import importlib.util

    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)

    src = tmp_path / "fake_models_src.py"
    src.write_text(
        "from tests.bg_fakes import FakeFactory\n"
        "print('noise-import')\n"
        "def preflight(settings):\n"
        "    print('noise-preflight')\n"
        "    return []\n"
        "def HFModelFactory(settings, class_names):\n"
        "    print('noise-factory')\n"
        "    return FakeFactory.default()\n", encoding="utf-8")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "background_swap.models":
                return importlib.util.spec_from_file_location(fullname, src)
            return None

    monkeypatch.delitem(sys.modules, "background_swap.models", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    assert main(["run", "--settings", str(p)]) == 0    # stream=None -> 실제 sys.stdout
    out, err = capsys.readouterr()
    for line in out.splitlines():
        json.loads(line)                                # stdout은 JSON Lines만
    assert "noise" not in out
    for marker in ("noise-import", "noise-preflight", "noise-factory"):
        assert marker in err


def test_progress_reaches_real_stdout_under_redirect(tmp_path, capsys):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default()) == 0
    out, _ = capsys.readouterr()
    ev = [json.loads(line) for line in out.splitlines()]
    assert any(e.get("stage") == "inpaint" for e in ev)
    assert ev[-1] == {"event": "done"}
