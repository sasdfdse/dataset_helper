import dataclasses
import io
import json
import sys

from background_swap.cli import main
from background_swap.common.domain import GpuOutOfMemory
from background_swap.common.settings import BackgroundSwapSettings, save_settings
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
    monkeypatch.setitem(sys.modules, "background_swap.models", None)    # ImportError on import
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], None, buf) == 5
    assert "make venv" in _events(buf)[-1]["message"]


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
    assert _events(buf)[-1]["skipped_unaccepted"] == 0


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
    assert main(["run", "--settings", str(p)]) == 0    # stream=None -> real sys.stdout
    out, err = capsys.readouterr()
    for line in out.splitlines():
        json.loads(line)                                # stdout carries JSON Lines only
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


def _korean_settings(tmp_path, **kw):
    from background_swap.common.settings import KeepItem
    classes = tmp_path / "classes.txt"
    classes.write_text("로봇\nball\n", encoding="utf-8")
    return _settings(tmp_path, n_images=2, use_vlm=False, use_llm=False, class_names_file=classes,
                     keep_items=[KeepItem(name="골대", method="text", prompt="골대")],
                     scene_axes={"venue": ["체육관"]}, **kw)


_KO_EN = {"골대": "goal post", "체육관": "gymnasium", "로봇": "robot"}


def test_preview_translates_korean_inputs_before_models(tmp_path):
    from tests.bg_fakes import FakeDetector, FakeTranslator
    s = _korean_settings(tmp_path)
    p = s.output_dir / "_preview_settings.json"
    save_settings(s, p)
    built, factories = [], []

    def builder(settings, class_names):
        built.append((settings, list(class_names)))
        f = FakeFactory.default(detector=FakeDetector({}), translator=FakeTranslator(_KO_EN))
        factories.append(f)
        return f

    buf = io.StringIO()
    assert main(["preview", "--settings", str(p), "--n", "1"], builder, buf) == 0
    ev = _events(buf)
    assert {"stage": "translate", "done": 3, "total": 3} in ev
    assert ev[-1]["translations"] == [
        {"kind": "keep", "source": "골대", "text": "goal post"},
        {"kind": "scene", "source": "체육관", "text": "gymnasium"},
        {"kind": "class", "source": "로봇", "text": "robot"}]
    # stages after translation use a factory built from the English settings
    final_settings, final_names = built[-1]
    assert final_settings.keep_items[0].prompt == "goal post"
    assert final_settings.scene_axes == {"venue": ["gymnasium"]}
    assert final_names == ["robot", "ball"]
    phrases = [ph for call in factories[-1].detector_obj.calls for ph in call[0]]
    assert "goal post" in phrases and "robot" in phrases
    assert not any("골대" in ph or "로봇" in ph for ph in phrases)
    assert "translator" not in factories[-1].opened
    # translations are saved in the output folder (not the preview folder) so Run reuses them
    cache = json.loads((s.output_dir / "translations.json").read_text(encoding="utf-8"))
    assert cache == _KO_EN
    prompts = [json.loads(line)["prompt"]
               for line in (s.output_dir / "_preview" / "jobs.jsonl").read_text().splitlines()]
    assert prompts and all("gymnasium" in pr and "체육관" not in pr for pr in prompts)

    run_p = s.output_dir / "run_settings.json"
    save_settings(s, run_p)
    built.clear()
    buf = io.StringIO()
    assert main(["run", "--settings", str(run_p)], builder, buf) == 0
    assert "translator" not in factories[-1].opened and all("translator" not in f.opened
                                                           for f in factories[-2:])
    assert built[-1][0].keep_items[0].prompt == "goal post"


def test_untranslatable_input_exits_2_with_the_text(tmp_path):
    from tests.bg_fakes import FakeTranslator
    s = _korean_settings(tmp_path)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    bad = dict(_KO_EN, 체육관="체육관")
    buf = io.StringIO()
    code = main(["run", "--settings", str(p)],
                lambda s, c: FakeFactory.default(translator=FakeTranslator(bad)), buf)
    assert code == 2
    last = _events(buf)[-1]
    assert last["event"] == "error" and "체육관" in last["message"] and "영어로 직접" in last["message"]


def test_preflight_checks_llm_when_translation_is_needed(tmp_path, monkeypatch):
    import types
    s = _korean_settings(tmp_path)
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    seen = []
    fake = types.ModuleType("background_swap.models")
    fake.HFModelFactory = lambda s, c: None
    fake.preflight = lambda s: (seen.append(s.use_llm), ["stop"])[1]
    monkeypatch.setitem(sys.modules, "background_swap.models", fake)
    assert main(["run", "--settings", str(p)], None, io.StringIO()) == 5
    assert seen == [True]
    # with saved translations, preflight checks the settings as-is (LLM off)
    (s.output_dir / "translations.json").write_text(json.dumps(_KO_EN), encoding="utf-8")
    seen.clear()
    assert main(["run", "--settings", str(p)], None, io.StringIO()) == 5
    assert seen == [False]


def test_unchecked_axes_are_not_used_or_translated(tmp_path):
    from tests.bg_fakes import FakeTranslator
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False, use_llm=False,
                  scene_axes={"venue": ["gym"], "camera": ["흐릿한 사진"]}, disabled_scene_axes=["camera"])
    p = s.output_dir / "run_settings.json"
    save_settings(s, p)
    tr = FakeTranslator()
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda s, c: FakeFactory.default(translator=tr), buf) == 0
    assert tr.calls == [] and "translations" not in _events(buf)[-1]
    jobs = [json.loads(line) for line in (s.output_dir / "jobs.jsonl").read_text().splitlines()]
    assert jobs and all(j["scene"] == {"venue": "gym"} for j in jobs)


def test_probe_reports_gpu_in_done_event(monkeypatch):
    import types
    fake = types.ModuleType("background_swap.models.probe")
    fake.probe_gpu = lambda: {"gpu": "Fake GPU", "vram_gb": 8.0, "tflops": 25.0}
    monkeypatch.setitem(sys.modules, "background_swap.models.probe", fake)
    buf = io.StringIO()
    assert main(["probe"], None, buf) == 0
    done = [e for e in _events(buf) if e.get("event") == "done"]
    assert done == [{"event": "done", "gpu": "Fake GPU", "vram_gb": 8.0, "tflops": 25.0}]


def test_reference_photos_add_axis_candidates_and_are_reported(tmp_path, monkeypatch):
    import types
    import cv2
    import numpy as np
    from tests.bg_fakes import FakeDescriber
    photo = tmp_path / "refs" / "light.jpg"
    photo.parent.mkdir()
    cv2.imwrite(str(photo), np.full((20, 30, 3), 90, np.uint8))
    s = _settings(tmp_path, n_images=1, num_variants=2, use_vlm=False, use_llm=False,
                  scene_axes={"venue": ["gym"], "lighting": []}, reference_images={"lighting": [photo]})
    p = s.output_dir / "_preview_settings.json"
    save_settings(s, p)
    d = FakeDescriber(["warm stage spotlights"])
    buf = io.StringIO()
    assert main(["preview", "--settings", str(p), "--n", "1"],
                lambda s, c: FakeFactory.default(describer=d), buf) == 0
    ev = _events(buf)
    assert {"stage": "references", "done": 1, "total": 1} in ev
    assert ev[-1]["references"] == [{"axis": "lighting", "file": "light.jpg", "text": "warm stage spotlights"}]
    jobs = [json.loads(line) for line in (s.output_dir / "_preview" / "jobs.jsonl").read_text().splitlines()]
    assert jobs and all(j["scene"]["lighting"] == "warm stage spotlights" for j in jobs)
    # reference captions need the VLM even when VLM review is off, so preflight checks it
    (s.output_dir / "references.json").unlink()
    seen = []
    fake = types.ModuleType("background_swap.models")
    fake.HFModelFactory = lambda s, c: None
    fake.preflight = lambda s: (seen.append(s.use_vlm), ["stop"])[1]
    monkeypatch.setitem(sys.modules, "background_swap.models", fake)
    assert main(["preview", "--settings", str(p)], None, io.StringIO()) == 5
    assert seen == [True]
