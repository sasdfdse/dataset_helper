import json

import cv2
import numpy as np
import pytest

from background_swap.store import imwrite_atomic, write_mask
from gui.background_results import (build_result_list, describe_record, format_export_message,
                                    format_report, index_originals, load_panels, load_report,
                                    result_image_path)

H, W = 40, 60


def make_output(tmp_path, with_files=True):
    """Manifest + 이미지 + 마스크를 가진 가짜 output_dir. 반환: (images, labels, out)."""
    images, labels, out = tmp_path / "images", tmp_path / "labels", tmp_path / "out"
    for d in (images, labels):
        d.mkdir()
    orig = np.full((H, W, 3), 100, np.uint8)
    cv2.imwrite(str(images / "a.jpg"), orig)
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
    acc = np.full((H, W, 3), 100, np.uint8)
    acc[~(protect | keep)] = 200
    imwrite_atomic(out / "Accepted" / "images" / "a_bgswap1.jpg", acc, 100)
    rej = acc.copy()
    rej[12:16, 22:26] = 0          # 보호 영역 내부 변경
    imwrite_atomic(out / "Rejected" / "a_bgswap2_a1.jpg", rej, 100)
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
    assert not red.any()
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
