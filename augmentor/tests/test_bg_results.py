import json

import cv2
import numpy as np
import pytest

from background_swap.stages.composite import diff_image
from background_swap.common.store import imwrite_atomic, write_mask
from gui.background_results import (DIFF_JPEG_NOTE, build_result_list, describe_record, format_export_message,
                                    format_report, index_originals, load_panels, load_report,
                                    result_image_path)

H, W = 40, 60


def make_output(tmp_path, with_files=True):
    """Manifest + 이미지 + 마스크를 가진 가짜 output_dir. 반환: (images, labels, out)."""
    images, labels, out = tmp_path / "images", tmp_path / "labels", tmp_path / "out"
    for d in (images, labels):
        d.mkdir()
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)      # 질감 있는 원본 (JPEG 오차가 드러나게)
    cv2.imwrite(str(images / "a.png"), orig)
    (labels / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    core = np.zeros((H, W), bool)
    core[10:30, 20:40] = True
    protect = np.zeros((H, W), bool)
    protect[8:32, 18:42] = True
    keep = np.zeros((H, W), bool)
    keep[0:5, :] = True
    for name, m in (("object_core", core), ("object_protect", protect), ("keep", keep),
                    ("inpaint", ~protect)):
        write_mask(out / "masks" / f"a_{name}.png", m)
    acc = orig.copy()                                  # 보호 영역은 원본 그대로, 바깥은 새 배경
    acc[~(protect | keep)] = rng.integers(0, 256, (int((~(protect | keep)).sum()), 3), dtype=np.uint8)
    rej = acc.copy()
    rej[12:16, 22:26] = (rej[12:16, 22:26].astype(np.int16) + 128).astype(np.uint8)   # 보호 영역 내부 변경
    for name, img in (("a_bgswap1_a1", acc), ("a_bgswap2_a1", rej)):
        (out / "generated").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "generated" / f"{name}.png"), img)      # 무손실 생성 결과
    imwrite_atomic(out / "Accepted" / "images" / "a_bgswap1.jpg", acc, 95)
    imwrite_atomic(out / "Rejected" / "a_bgswap2_a1.jpg", rej, 95)
    recs = [
        {"stem": "a", "variant": 1, "attempt": 1, "seed": 1, "scene": {"venue": "gym"},
         "prompt": "a gym", "status": "accepted", "reasons": [],
         "checks": {"protected_identical": True, "mean_change": 55.5, "gdino_new_count": 0,
                    "vlm": {"natural_indoor_venue": True, "boundary_artifacts": False,
                            "new_objects": False, "reason": "ok"}}, "warnings": []},
        {"stem": "a", "variant": 2, "attempt": 1, "seed": 2, "scene": {"venue": "hall"},
         "prompt": "a hall", "status": "rejected", "reasons": ["protected_changed"],
         "checks": {"protected_identical": False, "mean_change": 3.0}, "warnings": []},
        {"stem": "b", "variant": None, "attempt": None, "seed": None, "scene": None,
         "prompt": None, "status": "skipped", "reasons": ["missing_label"], "checks": {},
         "warnings": []},
        {"stem": "a", "variant": 3, "attempt": None, "status": "exhausted", "reasons": [],
         "checks": {}, "warnings": []},
    ]
    out.mkdir(exist_ok=True)
    (out / "Manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return images, labels, out


def test_build_result_list_filters_and_counts(tmp_path):
    _, _, out = make_output(tmp_path)
    rl = build_result_list(out)
    assert [(e.stem, e.variant, e.attempt, e.status) for e in rl.entries] == [
        ("a", 1, 1, "accepted"), ("a", 2, 1, "rejected")]
    assert rl.counts == {"accepted": 1, "rejected": 1, "skipped": 1, "exhausted": 1}
    assert "skipped 1" in rl.summary() and "accepted 1" in rl.summary()


def test_build_result_list_empty(tmp_path):
    assert build_result_list(tmp_path / "nope").entries == []


def test_result_image_path(tmp_path):
    _, _, out = make_output(tmp_path)
    rl = build_result_list(out)
    assert rl.entries[0].path == out / "Accepted" / "images" / "a_bgswap1.jpg"
    assert rl.entries[1].path == out / "Rejected" / "a_bgswap2_a1.jpg"
    assert result_image_path(out, {"status": "rejected", "stem": "x", "variant": 3, "attempt": 2}) \
        == out / "Rejected" / "x_bgswap3_a2.jpg"


def test_load_panels_accepted_has_no_red_in_diff(tmp_path):
    images, labels, out = make_output(tmp_path)
    rl = build_result_list(out)
    items = index_originals(images, labels)
    p = load_panels(out, rl.entries[0], items["a"])
    for img in (p.orig, p.overlay, p.result, p.diff):
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
    red = np.all(p.diff == (0, 0, 255), axis=2)
    assert not red.any() and p.diff_note == ""
    assert p.diff[35, 5].max() > 0          # 바깥은 변경됨 (회색 차이)
    assert not np.array_equal(p.overlay, p.orig)


def test_load_panels_rejected_shows_red_in_protected_area(tmp_path):
    images, labels, out = make_output(tmp_path)
    rl = build_result_list(out)
    items = index_originals(images, labels)
    p = load_panels(out, rl.entries[1], items["a"])
    assert np.all(p.diff == (0, 0, 255), axis=2).any()


def test_load_panels_missing_file_raises(tmp_path):
    images, labels, out = make_output(tmp_path)
    rl = build_result_list(out)
    (out / "Accepted" / "images" / "a_bgswap1.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        load_panels(out, rl.entries[0], index_originals(images, labels)["a"])


def test_describe_record():
    text = describe_record({"status": "rejected", "reasons": ["protected_changed", "vlm_unnatural"],
                            "checks": {"protected_identical": False, "mean_change": 3.0,
                                       "gdino_new_count": 2,
                                       "vlm": {"natural_indoor_venue": False, "reason": "dark"}},
                            "scene": {"venue": "gym", "light": "dim"}, "prompt": "a gym"})
    assert "protected_changed, vlm_unnatural" in text
    assert "venue=gym" in text and "light=dim" in text
    assert "a gym" in text and "mean_change" in text and "dark" in text
    # 누락/None 필드에도 예외 없음
    assert describe_record({"status": "accepted", "checks": None, "scene": None, "reasons": None})


def test_format_report_and_load_report(tmp_path):
    report = {"status_counts": {"accepted": 3, "rejected": 1},
              "reason_counts": {"protected_changed": 1},
              "scene_counts": {"all": {"venue": {"gym": 3, "hall": 1}},
                               "accepted": {"venue": {"gym": 3}}},
              "class_recall": {"ball": {"hits": 1, "total": 2, "recall": 0.5},
                               "cone": {"hits": 0, "total": 0, "recall": None}},
              "stage_seconds": {"masks": 1.5},
              "note": "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다."}
    text = format_report(report)
    for needle in ("accepted", "protected_changed", "venue", "gym", "ball", "50.0%", "n/a",
                   "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다.", "masks"):
        assert needle in text
    assert "평균 생성 시간" not in text
    assert "평균 생성 시간" not in format_report({**report, "avg_seconds_per_generation": None})
    text = format_report({**report, "avg_seconds_per_generation": 2.5})
    assert "[이미지당 평균 생성 시간]" in text and "2.50초" in text
    assert format_report({})        # 빈 리포트도 예외 없음
    assert load_report(tmp_path) is None
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert load_report(tmp_path) == report
    (tmp_path / "report.json").write_text("{broken")
    assert load_report(tmp_path) is None


def test_format_export_message():
    assert "3" in format_export_message({"event": "done", "copied": 3, "skipped_existing": 2})
    assert "2" in format_export_message({"event": "done", "copied": 3, "skipped_existing": 2})
    assert format_export_message({"event": "done"})
    assert "승인 기록" not in format_export_message(
        {"event": "done", "copied": 3, "skipped_existing": 2, "skipped_unaccepted": 0})
    msg = format_export_message({"event": "done", "copied": 3, "skipped_existing": 2,
                                 "skipped_unaccepted": 4})
    assert "승인 기록이 없어 건너뜀 4쌍" in msg


def _protect(out):
    from background_swap.common.store import read_mask
    return read_mask(out / "masks" / "a_object_protect.png") | read_mask(out / "masks" / "a_keep.png")


def test_diff_uses_lossless_generated_png(tmp_path):
    images, labels, out = make_output(tmp_path)
    rl = build_result_list(out)
    items = index_originals(images, labels)
    orig = cv2.imread(str(images / "a.png"))
    protect = _protect(out)
    # 질감 이미지에서 JPEG q95 결과와 직접 비교하면 보호 영역이 빨갛게 나온다 (이 테스트가 민감한지 확인)
    jpeg = cv2.imread(str(rl.entries[0].path))
    assert np.all(diff_image(orig, jpeg, protect) == (0, 0, 255), axis=2)[protect].mean() > 0.5
    p = load_panels(out, rl.entries[0], items["a"])
    red = np.all(p.diff == (0, 0, 255), axis=2)
    assert not red[protect].any() and p.diff_note == ""
    # 패널 3(결과)은 계속 JPEG 결과 파일
    assert np.array_equal(p.result, jpeg)


def test_diff_falls_back_to_jpeg_when_png_missing(tmp_path):
    images, labels, out = make_output(tmp_path)
    rl = build_result_list(out)
    (out / "generated" / "a_bgswap1_a1.png").unlink()
    p = load_panels(out, rl.entries[0], index_originals(images, labels)["a"])
    assert p.diff_note == DIFF_JPEG_NOTE and "JPEG" in p.diff_note
    assert p.diff.shape == (H, W, 3)
