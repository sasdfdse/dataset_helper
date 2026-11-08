"""models.py의 GPU 없이 검증 가능한 부분: oom_retry, match_phrase, 팩토리 로드 OOM 처리.

torch/transformers/diffusers가 없는 환경(시스템 python3)에서는 건너뛴다. importorskip은 모듈
최상단이 아니라 fixture 안에서 한다 — ROS launch_testing 플러그인이 있는 시스템 pytest 7에서는
모듈 수준 skip이 tests 패키지 전체 수집을 건너뛰게 만든다. 실제 모델은 로드하지 않는다.
"""
import importlib
import sys

import pytest

from background_swap.domain import GpuOutOfMemory
from background_swap.settings import BackgroundSwapSettings


@pytest.fixture
def M(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    # 다른 테스트(test_bg_cli)가 sys.modules["background_swap.models"]에 가짜 모듈을 넣고 남길 수
    # 있으므로 항목을 지우고 실제 모듈을 새로 import한다 (teardown에서 이전 상태로 복원됨).
    monkeypatch.delitem(sys.modules, "background_swap.models", raising=False)
    return importlib.import_module("background_swap.models")


def _oom(M):
    return M.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")


def _flaky(M, fails: int):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= fails:
            raise _oom(M)
        return "ok"
    return fn, calls


# ── oom_retry ──────────────────────────────────────────────────────

def test_oom_retry_returns_without_retry_on_success(M):
    fn, calls = _flaky(M, 0)
    assert M.oom_retry(fn) == "ok"
    assert len(calls) == 1


def test_oom_retry_retries_once_then_succeeds(M):
    fn, calls = _flaky(M, 1)
    assert M.oom_retry(fn) == "ok"
    assert len(calls) == 2


def test_oom_retry_raises_gpu_out_of_memory_after_second_oom(M):
    fn, calls = _flaky(M, 2)
    with pytest.raises(GpuOutOfMemory, match="fake"):
        M.oom_retry(fn)
    assert len(calls) == 2


def test_oom_retry_does_not_retry_other_errors(M):
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("not oom")
    with pytest.raises(ValueError):
        M.oom_retry(fn)
    assert len(calls) == 1


# ── match_phrase ───────────────────────────────────────────────────

def test_match_phrase_exact_case_insensitive(M):
    assert M.match_phrase("Red Box", ["robot", "red box"]) == "red box"
    assert M.match_phrase("robot.", ["Robot", "box"]) == "Robot"


def test_match_phrase_substring_either_direction(M):
    assert M.match_phrase("green floor", ["box", "floor"]) == "floor"
    assert M.match_phrase("box", ["robot", "red box"]) == "red box"


def test_match_phrase_falls_back_to_first(M):
    assert M.match_phrase("zebra", ["box", "floor"]) == "box"
    assert M.match_phrase("", ["box", "floor"]) == "box"


# ── 팩토리: 로드 중 OOM ────────────────────────────────────────────

class _Fake:
    """생성자가 앞의 fails번은 OOM을 던지는 가짜 래퍼. 생성/close 기록은 클래스 공유 log에."""

    def __init__(self, oom_exc, fails: int, log: list, name: str):
        self.oom_exc, self.fails, self.log, self.name = oom_exc, fails, log, name
        self.attempts = 0

    def __call__(self, *args):
        self.attempts += 1
        if self.attempts <= self.fails:
            raise self.oom_exc
        log, name = self.log, self.name

        class Obj:
            def close(self):
                log.append(("close", name))
        log.append(("init", name))
        return Obj()


_CONTEXTS = [
    ("prompt_model", "HFPromptGenerator"),
    ("inpainter", "HFInpainter"),
    ("detector", "HFTextDetector"),
    ("validator", "HFValidator"),
]


def _factory(M):
    return M.HFModelFactory(BackgroundSwapSettings(), ["box"])


@pytest.mark.parametrize("method,cls", _CONTEXTS)
def test_factory_load_oom_twice_raises_gpu_out_of_memory(M, monkeypatch, method, cls):
    log: list = []
    fake = _Fake(_oom(M), 2, log, cls)
    monkeypatch.setattr(M, cls, fake)
    with pytest.raises(GpuOutOfMemory):
        with getattr(_factory(M), method)():
            pytest.fail("context body must not run")
    assert fake.attempts == 2
    assert log == []


@pytest.mark.parametrize("method,cls", _CONTEXTS)
def test_factory_load_oom_once_then_yields_and_releases(M, monkeypatch, method, cls):
    log: list = []
    fake = _Fake(_oom(M), 1, log, cls)
    monkeypatch.setattr(M, cls, fake)
    with getattr(_factory(M), method)() as obj:
        assert hasattr(obj, "close")
        assert log == [("init", cls)]
    assert fake.attempts == 2
    assert log == [("init", cls), ("close", cls)]


def test_mask_models_load_oom_retry_yields_both(M, monkeypatch):
    log: list = []
    seg = _Fake(_oom(M), 1, log, "seg")
    det = _Fake(_oom(M), 1, log, "det")
    monkeypatch.setattr(M, "HFSegmenter", seg)
    monkeypatch.setattr(M, "HFTextDetector", det)
    with _factory(M).mask_models() as (s, d):
        assert s is not None and d is not None
    assert (seg.attempts, det.attempts) == (2, 2)
    assert log == [("init", "seg"), ("init", "det"), ("close", "seg"), ("close", "det")]


def test_mask_models_detector_oom_releases_loaded_segmenter(M, monkeypatch):
    log: list = []
    monkeypatch.setattr(M, "HFSegmenter", _Fake(_oom(M), 0, log, "seg"))
    monkeypatch.setattr(M, "HFTextDetector", _Fake(_oom(M), 2, log, "det"))
    with pytest.raises(GpuOutOfMemory):
        with _factory(M).mask_models():
            pytest.fail("context body must not run")
    assert log == [("init", "seg"), ("close", "seg")]


def test_factory_releases_on_exception_in_body(M, monkeypatch):
    log: list = []
    monkeypatch.setattr(M, "HFValidator", _Fake(_oom(M), 0, log, "vlm"))
    with pytest.raises(RuntimeError, match="body"):
        with _factory(M).validator():
            raise RuntimeError("body")
    assert log == [("init", "vlm"), ("close", "vlm")]
