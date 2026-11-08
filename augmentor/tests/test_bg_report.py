import json
import os

import pytest

from background_swap.common.domain import Detection
from background_swap.dataset import export as export_mod
from background_swap.dataset.export import ExportResult, NonTrainTargetError, export_accepted
from background_swap.common.progress import NullProgress
from background_swap.dataset.report import build_report, class_recall, merge_recall, write_report
from background_swap.common.store import append_jsonl, read_jsonl
from background_swap.pipeline.workflow import BackgroundSwapWorkflow
from tests.bg_fakes import *  # noqa: F401,F403

_settings = make_settings


def _rec(status, reasons=(), scene=None):
    return {"stem": "a", "variant": 1, "attempt": 1, "seed": 1, "scene": scene, "prompt": "p",
            "status": status, "reasons": list(reasons), "checks": {}, "warnings": []}


def _accepted(tmp_path, stems, record=True):
    """Accepted/에 `{stem}_bgswap{n}` 쌍을 만들고 (record면) Manifest에 accepted 레코드를 쓴다."""
    out = tmp_path / "out"
    (out / "Accepted" / "images").mkdir(parents=True)
    (out / "Accepted" / "labels").mkdir(parents=True)
    for s in stems:
        (out / "Accepted" / "images" / f"{s}.jpg").write_bytes(b"img-" + s.encode())
        (out / "Accepted" / "labels" / f"{s}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        if record:
            stem, n = s.rsplit("_bgswap", 1)
            append_jsonl(out / "Manifest.jsonl", {**_rec("accepted"), "stem": stem,
                                                   "variant": int(n)})
    return out


def test_class_recall():
    labels = [(0, (0, 0, 10, 10)), (0, (50, 50, 60, 60)), (1, (20, 20, 30, 30))]
    dets = [Detection((0, 0, 10, 9), 0.9, "robot"), Detection((20, 20, 30, 30), 0.9, "robot")]
    assert class_recall(labels, dets, ["robot", "ball"]) == {"robot": [1, 2], "ball": [0, 1]}


def test_class_recall_ignores_out_of_range_cls():
    assert class_recall([(5, (0, 0, 1, 1)), (-1, (0, 0, 1, 1))], [], ["robot"]) == {"robot": [0, 0]}


def test_merge_recall():
    assert merge_recall({"a": [1, 2]}, {"a": [1, 1], "b": [0, 3]}) == {"a": [2, 3], "b": [0, 3]}


def test_build_report_counts(tmp_path):
    for r in [_rec("accepted", scene={"venue": "gym"}),
              _rec("rejected", ["vlm_unnatural"], scene={"venue": "hall"}),
              _rec("skipped", ["small_inpaint_area"])]:
        append_jsonl(tmp_path / "Manifest.jsonl", r)
    rep = build_report(tmp_path)
    assert rep["status_counts"] == {"accepted": 1, "rejected": 1, "skipped": 1}
    assert rep["reason_counts"] == {"vlm_unnatural": 1, "small_inpaint_area": 1}
    assert rep["scene_counts"]["accepted"] == {"venue": {"gym": 1}}
    assert rep["scene_counts"]["all"] == {"venue": {"gym": 1, "hall": 1}}
    assert "class_recall" not in rep and "stage_seconds" not in rep
    assert rep["note"] == "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다."


def test_build_report_recall_and_timings(tmp_path):
    (tmp_path / "recall.json").write_text(json.dumps({"robot": [1, 3], "ball": [0, 0]}))
    (tmp_path / "timings.json").write_text(json.dumps({"masks": 1.5}))
    rep = build_report(tmp_path)
    assert rep["class_recall"] == {"robot": {"hits": 1, "total": 3, "recall": 0.3333},
                                   "ball": {"hits": 0, "total": 0, "recall": None}}
    assert rep["stage_seconds"] == {"masks": 1.5}
    assert rep["status_counts"] == {} and rep["reason_counts"] == {}


def test_write_report_writes_json(tmp_path):
    rep = write_report(tmp_path)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == rep


def test_export_copies_and_skips_existing(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1", "a_bgswap2"])
    imgs, lbls = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    imgs.mkdir(parents=True); lbls.mkdir()
    (imgs / "a_bgswap2.jpg").write_bytes(b"old")
    assert export_accepted(out, imgs, lbls, False) == ExportResult(copied=1, skipped_existing=1)
    assert (imgs / "a_bgswap2.jpg").read_bytes() == b"old"
    assert (imgs / "a_bgswap1.jpg").read_bytes() == b"img-a_bgswap1"
    assert not (lbls / "a_bgswap2.txt").exists()


def test_export_skips_pair_when_only_label_exists(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    imgs, lbls = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    lbls.mkdir(parents=True)
    (lbls / "a_bgswap1.txt").write_text("old")
    assert export_accepted(out, imgs, lbls, False) == ExportResult(0, 1)
    assert not (imgs / "a_bgswap1.jpg").exists()


def test_export_creates_dirs_and_ignores_unpaired_images(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    (out / "Accepted" / "images" / "b_bgswap1.jpg").write_bytes(b"x")   # 라벨 없음
    imgs, lbls = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    assert export_accepted(out, imgs, lbls, False) == ExportResult(1, 0)
    assert [p.name for p in imgs.iterdir()] == ["a_bgswap1.jpg"]


def test_export_skips_pairs_without_accepted_manifest_record(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    _accepted_extra = out / "Accepted"
    for name in ("b_bgswap1", "a_bgswap2"):      # 파일만 있고 accepted 레코드 없음 (고아)
        (_accepted_extra / "images" / f"{name}.jpg").write_bytes(b"orphan")
        (_accepted_extra / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    append_jsonl(out / "Manifest.jsonl", {**_rec("rejected"), "stem": "b"})
    imgs, lbls = _dest(tmp_path)
    r = export_accepted(out, imgs, lbls, False)
    assert r == ExportResult(copied=1, skipped_existing=0, skipped_unaccepted=2)
    assert [p.name for p in imgs.iterdir()] == ["a_bgswap1.jpg"]
    assert [p.name for p in lbls.iterdir()] == ["a_bgswap1.txt"]


def test_export_without_manifest_copies_nothing(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"], record=False)
    imgs, lbls = _dest(tmp_path)
    assert export_accepted(out, imgs, lbls, False) == ExportResult(0, 0, 1)
    assert list(imgs.iterdir()) == []


def test_export_result_positional_compat():
    r = ExportResult(2, 1)
    assert (r.copied, r.skipped_existing, r.skipped_unaccepted) == (2, 1, 0)


def test_export_orphan_after_crash_then_reject_on_resume(tmp_path):
    """Accepted 파일 기록 후 Manifest 기록 전에 중단 → 재개 시 거절: 내보내지 않고 고아 파일도 지운다."""
    s = _settings(tmp_path, n_images=1, use_llm=False, num_variants=1, max_attempts=1)

    class Crash(Exception):
        pass

    wf = BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([vlm_json(True)])),
                                NullProgress(), [])
    orig = wf._append_manifest

    def boom(rec):
        if rec["status"] == "accepted":
            raise Crash()
        orig(rec)
    wf._append_manifest = boom
    with pytest.raises(Crash):
        wf.run()
    acc = s.output_dir / "Accepted"
    stale = [acc / "images" / "img_000_bgswap1.jpg", acc / "labels" / "img_000_bgswap1.txt",
             acc / "previews" / "img_000_bgswap1.jpg"]
    assert all(p.exists() for p in stale)
    # 미리 만든 Accepted 파일로 export 필터도 확인 (Manifest에 accepted 없음)
    imgs, lbls = _dest(tmp_path / "pre")
    assert export_accepted(s.output_dir, imgs, lbls, False) == ExportResult(0, 0, 1)

    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([vlm_json(False)])),
                           NullProgress(), []).run()
    statuses = [r["status"] for r in read_jsonl(s.output_dir / "Manifest.jsonl")]
    assert statuses == ["rejected", "exhausted"]
    assert not any(p.exists() for p in stale)
    imgs, lbls = _dest(tmp_path / "post")
    assert export_accepted(s.output_dir, imgs, lbls, False) == ExportResult(0, 0, 0)
    assert list(imgs.iterdir()) == []


def test_export_orphan_counted_when_files_survive(tmp_path):
    """reject 경로를 거치지 않은 고아(예: 수동 복사)도 export가 걸러내고 skipped_unaccepted 1."""
    s = _settings(tmp_path, n_images=1, use_llm=False, num_variants=1, max_attempts=1)
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([vlm_json(False)])),
                           NullProgress(), []).run()
    acc = s.output_dir / "Accepted"
    (acc / "images").mkdir(parents=True, exist_ok=True)
    (acc / "labels").mkdir(parents=True, exist_ok=True)
    (acc / "images" / "img_000_bgswap1.jpg").write_bytes(b"orphan")
    (acc / "labels" / "img_000_bgswap1.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    imgs, lbls = _dest(tmp_path / "dst")
    r = export_accepted(s.output_dir, imgs, lbls, False)
    assert (r.copied, r.skipped_unaccepted) == (0, 1)
    assert list(imgs.iterdir()) == [] and list(lbls.iterdir()) == []


def test_reject_keeps_accepted_files_of_other_variant(tmp_path):
    """거절 시 정리는 같은 (stem, variant)의 고아만 — 다른 변형의 정상 Accepted 파일은 그대로."""
    s = _settings(tmp_path, n_images=1, use_llm=False, num_variants=2, max_attempts=1)
    # 변형 1 승인, 변형 2 거절 (VLM 응답 순서)
    BackgroundSwapWorkflow(s, FakeFactory.default(
        validator=FakeValidator([vlm_json(True), vlm_json(False)])), NullProgress(), []).run()
    acc = s.output_dir / "Accepted"
    assert (acc / "images" / "img_000_bgswap1.jpg").exists()
    assert not (acc / "images" / "img_000_bgswap2.jpg").exists()
    imgs, lbls = _dest(tmp_path / "dst")
    assert export_accepted(s.output_dir, imgs, lbls, False) == ExportResult(1, 0, 0)


def test_export_refuses_val_target(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    with pytest.raises(NonTrainTargetError):
        export_accepted(out, tmp_path / "val" / "images", tmp_path / "val" / "labels", False)
    assert not (tmp_path / "val").exists()
    assert export_accepted(out, tmp_path / "val" / "images", tmp_path / "val" / "labels",
                           True).copied == 1


def _dest(tmp_path):
    imgs, lbls = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    imgs.mkdir(parents=True); lbls.mkdir()
    return imgs, lbls


@pytest.mark.parametrize("fail_at", ["label_copy", "label_replace"])
def test_export_failure_mid_pair_leaves_no_half_pair(tmp_path, monkeypatch, fail_at):
    out = _accepted(tmp_path, ["a_bgswap1"])
    imgs, lbls = _dest(tmp_path)
    if fail_at == "label_copy":
        real = export_mod.shutil.copyfile

        def bad_copy(src, dst, **kw):
            if str(src).endswith(".txt"):
                raise OSError("disk full")
            return real(src, dst, **kw)
        monkeypatch.setattr(export_mod.shutil, "copyfile", bad_copy)
    else:
        real = os.replace

        def bad_replace(src, dst):
            if str(dst).endswith(".txt"):
                raise OSError("boom")
            return real(src, dst)
        monkeypatch.setattr(export_mod.os, "replace", bad_replace)
    with pytest.raises(OSError):
        export_accepted(out, imgs, lbls, False)
    assert list(imgs.iterdir()) == [] and list(lbls.iterdir()) == []   # 임시 파일 포함 아무것도 없음
    monkeypatch.undo()
    assert export_accepted(out, imgs, lbls, False) == ExportResult(1, 0)
    assert [p.name for p in imgs.iterdir()] == ["a_bgswap1.jpg"]
    assert [p.name for p in lbls.iterdir()] == ["a_bgswap1.txt"]


def test_export_files_have_normal_permissions(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    imgs, lbls = _dest(tmp_path)
    export_accepted(out, imgs, lbls, False)
    um = os.umask(0); os.umask(um)
    for p in (imgs / "a_bgswap1.jpg", lbls / "a_bgswap1.txt"):
        assert p.stat().st_mode & 0o777 == 0o666 & ~um


def test_workflow_writes_report(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    rep = json.loads((s.output_dir / "report.json").read_text())
    assert rep["status_counts"] == {"accepted": 1}
    assert set(rep["stage_seconds"]) >= {"masks", "prompts", "inpaint", "validate"}
    t = json.loads((s.output_dir / "timings.json").read_text())
    assert "report" in t
