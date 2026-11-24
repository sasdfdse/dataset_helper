"""GPU 확인: 이름, VRAM, fp16 행렬곱 속도 (모델은 올리지 않는다, 1~2초).

앱을 켤 때 GUI가 워커의 `probe` 명령으로 실행해 PC마다 예상 시간과 모델 적재 방식을 맞춘다.
"""
from __future__ import annotations

import time

_N = 4096
_WARMUP_S = 0.4
_MEASURE_S = 0.6


def probe_gpu() -> dict:
    """{"gpu": 이름 | None, "vram_gb": float, "tflops": float}"""
    import torch
    if not torch.cuda.is_available():
        return {"gpu": None, "vram_gb": 0.0, "tflops": 0.0}
    props = torch.cuda.get_device_properties(0)
    a = torch.randn(_N, _N, device="cuda", dtype=torch.float16)
    b = torch.randn(_N, _N, device="cuda", dtype=torch.float16)

    def burst(seconds: float) -> tuple[int, float]:
        n, start = 0, time.perf_counter()
        while True:
            for _ in range(4):
                a @ b
            torch.cuda.synchronize()
            n += 4
            elapsed = time.perf_counter() - start
            if elapsed >= seconds:
                return n, elapsed

    burst(_WARMUP_S)                 # until clocks ramp up (laptop GPUs start slow)
    n, elapsed = burst(_MEASURE_S)
    tflops = 2 * _N ** 3 * n / elapsed / 1e12
    del a, b
    torch.cuda.empty_cache()
    return {"gpu": props.name, "vram_gb": round(props.total_memory / 2 ** 30, 2),
            "tflops": round(tflops, 2)}
