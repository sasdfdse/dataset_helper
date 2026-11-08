import json

import cv2
import numpy as np
import pytest

from background_swap.stages.composite import draw_boxes
from background_swap.common.domain import Detection
from background_swap.common.progress import NullProgress
from background_swap.stages.prompts import PromptRequest, TemplatePromptGenerator, sanitize_prompt
from background_swap.common.settings import KeepItem
from background_swap.common.store import read_jsonl
from background_swap.pipeline.workflow import BackgroundSwapWorkflow, run_preview
from core.label import BBox
from tests.bg_fakes import *  # noqa: F401,F403

_settings = make_settings
_vlm = vlm_json


def test_run_produces_accepted_with_original_labels(tmp_path):
    s = _settings(tmp_path, n_images=2, num_variants=2, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    acc = sorted(p.name for p in (s.output_dir/"Accepted"/"images").iterdir())
    assert acc == ["img_000_bgswap1.jpg", "img_000_bgswap2.jpg", "img_001_bgswap1.jpg", "img_001_bgswap2.jpg"]
    assert (s.output_dir/"Accepted"/"labels"/"img_000_bgswap1.txt").read_text() == (s.labels_dir/"img_000.txt").read_text()


def test_rejected_then_retried_until_exhausted(tmp_path):
    bad = '{"natural_indoor_venue": false, "boundary_artifacts": false, "new_objects": false}'
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=3)
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([bad])), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [r["status"] for r in m] == ["rejected", "rejected", "rejected", "exhausted"]
    assert len({r["seed"] for r in m if r["status"] == "rejected"}) == 3
    assert len(list((s.output_dir/"Rejected").iterdir())) == 3


def test_retry_stops_after_accept(tmp_path):
    bad, good = _vlm(False), _vlm(True)
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=3)
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([bad, good])), NullProgress(), []).run()
    assert [r["status"] for r in read_jsonl(s.output_dir/"Manifest.jsonl")] == ["rejected", "accepted"]


def test_resume_skips_completed_work(tmp_path):
    s = _settings(tmp_path, n_images=2, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    f2 = FakeFactory.default()
    BackgroundSwapWorkflow(s, f2, NullProgress(), []).run()
    assert f2.inpainter_obj.calls == 0 and f2.opened == []
    assert len(read_jsonl(s.output_dir/"Manifest.jsonl")) == 2


def test_resume_after_crash_mid_inpaint(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=1, use_vlm=False)
    crashing = FakeFactory.default(inpainter=FakeInpainter(fail_after=1))
    with pytest.raises(RuntimeError):
        BackgroundSwapWorkflow(s, crashing, NullProgress(), []).run()
    f2 = FakeFactory.default()
    BackgroundSwapWorkflow(s, f2, NullProgress(), []).run()
    assert f2.inpainter_obj.calls == 2
    assert len(list((s.output_dir/"Accepted"/"images").iterdir())) == 3


def test_skipped_items_recorded_once(tmp_path):
    s = _settings(tmp_path, n_images=1, keep_items=[KeepItem("fence", "text", required=True, prompt="fence")])
    for _ in range(2):
        BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [(r["status"], r["reasons"]) for r in m] == [("skipped", ["required_keep_missing:fence"])]


def test_class_names_enable_gdino_new_object_check(tmp_path):
    det = FakeDetector({"robot": [Detection((1, 1, 5, 5), 0.9, "robot")]})   # 배경 영역 새 객체
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(detector=det), NullProgress(), ["robot"]).run()
    r = read_jsonl(s.output_dir/"Manifest.jsonl")[0]
    assert r["status"] == "rejected" and "new_object:gdino" in r["reasons"]


def test_use_llm_false_does_not_open_prompt_model(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_llm=False, use_vlm=False)
    f = FakeFactory.default(); BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert "prompt_model" not in f.opened and "validator" not in f.opened


def test_each_model_context_opened_once_per_round(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=2, max_attempts=1)
    f = FakeFactory.default(); BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert f.opened == ["mask_models", "prompt_model", "inpainter", "validator"]


def test_rgba_input_outputs_three_channel_jpg(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False, rgba=True)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    out = cv2.imread(str(s.output_dir/"Accepted"/"images"/"img_000_bgswap1.jpg"), cv2.IMREAD_UNCHANGED)
    assert out.shape[2] == 3


def test_run_preview_uses_isolated_dir(tmp_path):
    s = _settings(tmp_path, n_images=5, num_variants=3)
    d = run_preview(s, FakeFactory.default(), NullProgress(), [], n=2, rng_seed=0)
    assert d == s.output_dir/"_preview"
    assert len(read_jsonl(d/"Manifest.jsonl")) == 2 and not (s.output_dir/"Manifest.jsonl").exists()


# --- 추가 검증 (구현 결정 고정) ---

def test_resume_after_crash_in_later_round_continues_rounds(tmp_path):
    """라운드 1이 끝난 뒤 크래시 → 재실행 시 라운드 1은 대상이 없어도 라운드 2를 계속한다."""
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=2)
    bad = _vlm(False)
    crashing = FakeFactory.default(inpainter=FakeInpainter(fail_after=1), validator=FakeValidator([bad]))
    with pytest.raises(RuntimeError):
        BackgroundSwapWorkflow(s, crashing, NullProgress(), []).run()
    assert [r["status"] for r in read_jsonl(s.output_dir/"Manifest.jsonl")] == ["rejected"]
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([bad])), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [(r["status"], r["attempt"]) for r in m] == [("rejected", 1), ("rejected", 2), ("exhausted", None)]


def test_manifest_record_schema(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    r = read_jsonl(s.output_dir/"Manifest.jsonl")[0]
    assert set(r) == {"stem", "variant", "attempt", "seed", "scene", "prompt", "status",
                      "reasons", "checks", "warnings"}
    assert r["status"] == "accepted" and r["variant"] == 1 and r["attempt"] == 1
    assert set(r["scene"]) == set(s.scene_axes) and r["prompt"]
    assert (s.output_dir/"generated"/"img_000_bgswap1_a1.png").exists()
    assert (s.output_dir/"Accepted"/"previews"/"img_000_bgswap1.jpg").exists()
    jobs = read_jsonl(s.output_dir/"jobs.jsonl")
    assert [(j["stem"], j["variant"], j["attempt"], j["seed"]) for j in jobs] == [("img_000", 1, 1, r["seed"])]


def test_skipped_record_fields(tmp_path):
    s = _settings(tmp_path, n_images=1, keep_items=[KeepItem("fence", "text", required=True, prompt="fence")])
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    r = read_jsonl(s.output_dir/"Manifest.jsonl")[0]
    assert (r["variant"], r["attempt"], r["seed"], r["scene"], r["prompt"], r["checks"]) == (
        None, None, None, None, None, {})
    assert not (s.output_dir/"masks"/"img_000_inpaint.png").exists()
    assert (s.output_dir/"masks"/"img_000.json").exists()


def test_vlm_prompt_includes_class_names(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1)
    v = FakeValidator([_vlm(True)])
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=v), NullProgress(), ["robot", "ball"]).run()
    assert len(v.calls) == 1 and "robot, ball" in v.calls[0][1]


def test_draw_boxes_returns_copy_with_class_color():
    img = np.zeros((20, 20, 3), np.uint8)
    out = draw_boxes(img, [BBox(0, 0.5, 0.5, 0.5, 0.5)])
    assert img.sum() == 0
    assert tuple(out[5, 10]) == (0x44, 0x44, 0xFF)   # "#FF4444" → BGR


class _ShortValidator(FakeValidator):
    """이미지 수보다 1개 적은 응답을 반환하는 고장난 검증기."""

    def judge_batch(self, images_rgb, prompt):
        return super().judge_batch(images_rgb, prompt)[:-1]


class _ShortPromptGen:
    def generate_batch(self, requests):
        return ["a gym"] * (len(requests) - 1)


def test_short_vlm_response_list_raises(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=1, max_attempts=1)
    f = FakeFactory.default(validator=_ShortValidator([_vlm(True)]))
    with pytest.raises(ValueError, match="validate"):
        BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert read_jsonl(s.output_dir/"Manifest.jsonl") == []
    assert not (s.output_dir/"Accepted").exists()


def test_short_prompt_response_list_raises(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=1, max_attempts=1, use_vlm=False)
    f = FakeFactory.default(prompt_gen=_ShortPromptGen())
    with pytest.raises(ValueError, match="prompts"):
        BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert read_jsonl(s.output_dir/"jobs.jsonl") == []
    assert read_jsonl(s.output_dir/"Manifest.jsonl") == []


class _FixedPromptGen:
    def __init__(self, text):
        self.text = text

    def generate_batch(self, requests):
        return [self.text] * len(requests)


# 두 번째 출력은 밑줄/콜론이 없어 settings.scene_axes가 정확히 전달돼야만 걸러진다.
@pytest.mark.parametrize("bad", ["Surroundings_floor: x", "venue, lighting"])
def test_keyword_llm_output_falls_back_to_template(tmp_path, bad):
    s = _settings(tmp_path, n_images=2, num_variants=2, use_llm=True, use_vlm=False)
    f = FakeFactory.default(prompt_gen=_FixedPromptGen(bad))
    BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert "prompt_model" in f.opened
    jobs = read_jsonl(s.output_dir/"jobs.jsonl")
    assert len(jobs) == 4
    for j in jobs:
        tpl = TemplatePromptGenerator().generate_batch([PromptRequest(j["scene"], [], j["seed"])])[0]
        assert j["prompt"] == sanitize_prompt(tpl, [])
        assert j["prompt"] != bad
    assert len({j["prompt"] for j in jobs}) > 1   # 장면별로 각자의 템플릿 사용


def test_llm_output_class_names_removed_in_workflow(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_llm=True, use_vlm=False)
    f = FakeFactory.default(prompt_gen=_FixedPromptGen(
        "A bright sports hall with a Robot arena marking and tidy banners."))
    BackgroundSwapWorkflow(s, f, NullProgress(), ["robot"]).run()
    jobs = read_jsonl(s.output_dir/"jobs.jsonl")
    assert len(jobs) == 1
    assert "robot" not in jobs[0]["prompt"].lower()
    assert jobs[0]["prompt"] == "A bright sports hall with a arena marking and tidy banners."


def test_vlm_pairing_across_batch_boundary(tmp_path):
    ok, bad = _vlm(True), _vlm(False)
    s = _settings(tmp_path, n_images=5, num_variants=1, max_attempts=1)
    v = FakeValidator([ok, ok, ok, ok, bad])
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=v), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [n for n, _ in v.calls] == [4, 1]
    jobs = [(r["stem"], r["status"]) for r in m if r["status"] in ("accepted", "rejected")]
    assert jobs == [("img_000", "accepted"), ("img_001", "accepted"), ("img_002", "accepted"),
                    ("img_003", "accepted"), ("img_004", "rejected")]


# --- 리포트용 recall / timings ---

def _robot_det():
    # 라벨 "0 0.5 0.5 0.3 0.3" @ 64x48 → xyxy (22, 17, 41, 31)
    return FakeDetector({"robot": [Detection((22, 17, 41, 31), 0.9, "robot")]})


def test_recall_written_once_and_reuses_mask_context(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_vlm=False)
    f = FakeFactory.default(detector=_robot_det())
    BackgroundSwapWorkflow(s, f, NullProgress(), ["robot"]).run()
    assert json.loads((s.output_dir/"recall.json").read_text()) == {"robot": [1, 1]}
    assert f.opened.count("mask_models") == 1
    # "detector" 컨텍스트는 validate 단계의 1회뿐 (recall용 추가 컨텍스트 없음)
    assert f.opened.count("detector") == 1
    f2 = FakeFactory.default(detector=_robot_det())
    BackgroundSwapWorkflow(s, f2, NullProgress(), ["robot"]).run()
    assert f2.opened == []
    assert json.loads((s.output_dir/"recall.json").read_text()) == {"robot": [1, 1]}


def test_recall_opens_detector_when_masks_already_done(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    assert not (s.output_dir/"recall.json").exists()
    f2 = FakeFactory.default(detector=_robot_det())
    BackgroundSwapWorkflow(s, f2, NullProgress(), ["robot"]).run()
    assert json.loads((s.output_dir/"recall.json").read_text()) == {"robot": [1, 1]}
    # 결과가 모두 끝난 상태라 validate는 열리지 않음 → recall용 detector 1회뿐
    assert f2.opened == ["detector"]


def test_no_recall_without_class_names(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    assert not (s.output_dir/"recall.json").exists()


def test_timings_accumulate_across_runs(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    t1 = json.loads((s.output_dir/"timings.json").read_text())
    assert set(t1) >= {"masks", "prompts", "inpaint", "validate", "report"}
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    t2 = json.loads((s.output_dir/"timings.json").read_text())
    assert all(t2[k] >= t1[k] for k in t1)


@pytest.mark.parametrize("label", ["0 0.5 0.5 0.3 0.3 0.91\n",            # 신뢰도 열 (6필드)
                                   "0 0.5 0.5 0.3 0.3\n0 0.1 0.1\n",       # 일부 줄만 잘못됨
                                   "0.0 0.5 0.5 0.3 0.3\n",                # 실수형 클래스
                                   "0 0.5 abc 0.3 0.3\n"])                 # 숫자가 아님
def test_unsupported_label_format_skipped_and_run_continues(tmp_path, label):
    s = _settings(tmp_path, n_images=2, use_llm=False, num_variants=1, max_attempts=1)
    (s.labels_dir / "img_000.txt").write_text(label)
    for _ in range(2):     # 재실행해도 스킵 레코드는 한 번만
        BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    m = read_jsonl(s.output_dir / "Manifest.jsonl")
    assert [(r["stem"], r["status"], r["reasons"]) for r in m] == [
        ("img_000", "skipped", ["unsupported_label_format"]),
        ("img_001", "accepted", [])]
    assert sorted(p.name for p in (s.output_dir / "Accepted" / "images").iterdir()) == [
        "img_001_bgswap1.jpg"]


def test_blank_lines_in_label_are_not_unsupported(tmp_path):
    s = _settings(tmp_path, n_images=1, use_llm=False, num_variants=1, max_attempts=1)
    (s.labels_dir / "img_000.txt").write_text("\n0 0.5 0.5 0.3 0.3\n   \n")
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    m = read_jsonl(s.output_dir / "Manifest.jsonl")
    assert [r["status"] for r in m] == ["accepted"]
