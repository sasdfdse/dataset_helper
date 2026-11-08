"""실제 HF 모델로 전체 파이프라인 1장 실행 (CUDA + 캐시된 모델 필요).

실행: python -m pytest -m gpu -v -s tests/test_bg_gpu_smoke.py
기본 실행(pytest.ini의 -m "not gpu")에서는 제외된다.

무거운 의존성 확인(importorskip)은 모듈 최상단이 아니라 테스트 안에서 한다:
ROS launch_testing 플러그인이 있는 시스템 pytest 7에서는 모듈 수준 skip이
tests 패키지 전체의 수집을 건너뛰게 만들기 때문이다.
"""
import time

import pytest

from background_swap.common.progress import NullProgress
from background_swap.common.store import read_jsonl
from background_swap.pipeline.workflow import BackgroundSwapWorkflow
from tests.bg_fakes import make_settings

pytestmark = pytest.mark.gpu


def test_full_pipeline_one_image_on_gpu(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from background_swap.models import HFModelFactory

    s = make_settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, size=(512, 384))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    BackgroundSwapWorkflow(s, HFModelFactory(s, ["box"]), NullProgress(), ["box"]).run()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()

    # 거부되면 워크플로가 시도 레코드 뒤에 exhausted 레코드를 덧붙이므로 마지막 "시도" 레코드를 본다
    records = read_jsonl(s.output_dir / "Manifest.jsonl")
    attempts = [r for r in records if r.get("attempt") is not None]
    assert attempts, records
    rec = attempts[-1]
    print(f"\nstatus={rec['status']} reasons={rec['reasons']}")
    print(f"prompt={rec['prompt']!r}")
    print(f"checks={rec['checks']}")
    print(f"peak_alloc={peak / 1024**3:.2f}GB "
          f"peak_reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f}GB wall={elapsed:.1f}s")

    assert rec["status"] in ("accepted", "rejected")
    assert rec["status"] == "accepted" or rec["reasons"]
    assert peak < 8 * 1024**3
