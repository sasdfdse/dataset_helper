import io
import json

import cv2
import numpy as np
import pytest

from background_swap.common.progress import JsonLinesProgress, NullProgress
from background_swap.common.store import (
    append_jsonl, atomic_write_bytes, atomic_write_text, imwrite_atomic,
    read_jsonl, read_mask, reset_output_dir, write_mask,
)


def test_read_jsonl_ignores_truncated_last_line(tmp_path):
    p = tmp_path/"m.jsonl"
    append_jsonl(p, {"a": 1}); append_jsonl(p, {"a": 2})
    with open(p, "a") as f: f.write('{"a": 3, "b"')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_append_after_truncated_line_is_readable(tmp_path):
    p = tmp_path/"m.jsonl"
    with open(p, "w") as f: f.write('{"a": 1}\n{"trunc')
    append_jsonl(p, {"a": 2})
    assert {"a": 2} in read_jsonl(p)


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path/"nope.jsonl") == []


def test_read_jsonl_ignores_bad_middle_blank_and_non_object_lines(tmp_path):
    p = tmp_path/"m.jsonl"
    p.write_text('{"a": 1}\nnot json\n\n[1, 2]\n"s"\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_append_jsonl_keeps_korean_and_creates_parent(tmp_path):
    p = tmp_path/"sub"/"m.jsonl"
    append_jsonl(p, {"msg": "안녕"})
    assert "안녕" in p.read_text(encoding="utf-8")
    assert read_jsonl(p) == [{"msg": "안녕"}]


def test_mask_roundtrip(tmp_path):
    m = np.zeros((4, 5), bool); m[1:3, 2:4] = True
    write_mask(tmp_path/"m.png", m)
    assert (read_mask(tmp_path/"m.png") == m).all()
    raw = cv2.imread(str(tmp_path/"m.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(raw).tolist()) == {0, 255}
    assert read_mask(tmp_path/"m.png").dtype == bool


def test_atomic_write_leaves_no_temp_files(tmp_path):
    atomic_write_text(tmp_path/"x"/"a.txt", "hi")
    assert [p.name for p in (tmp_path/"x").iterdir()] == ["a.txt"]


def test_atomic_write_bytes_overwrites(tmp_path):
    atomic_write_bytes(tmp_path/"b.bin", b"one")
    atomic_write_bytes(tmp_path/"b.bin", b"two")
    assert (tmp_path/"b.bin").read_bytes() == b"two"
    assert [p.name for p in tmp_path.iterdir()] == ["b.bin"]


def test_imwrite_atomic_png_and_jpg(tmp_path):
    img = np.full((8, 6, 3), 120, np.uint8)
    imwrite_atomic(tmp_path/"a.png", img)
    imwrite_atomic(tmp_path/"b.jpg", img, jpeg_quality=90)
    assert cv2.imread(str(tmp_path/"a.png")).shape == (8, 6, 3)
    assert cv2.imread(str(tmp_path/"b.jpg")).shape == (8, 6, 3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "b.jpg"]


def test_imwrite_atomic_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        imwrite_atomic(tmp_path/"a.xyz", np.zeros((2, 2, 3), np.uint8))
    assert list(tmp_path.iterdir()) == []


def test_reset_output_dir_keeps_unrelated(tmp_path):
    (tmp_path/"masks").mkdir(); (tmp_path/"Manifest.jsonl").write_text("")
    (tmp_path/"notes.txt").write_text("keep")
    reset_output_dir(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]


def test_reset_output_dir_removes_run_artifacts_but_not_preview(tmp_path):
    for d in ("masks", "generated", "Accepted", "Rejected", "_preview"):
        (tmp_path/d).mkdir(); (tmp_path/d/"f.png").write_text("x")
    for f in ("items.jsonl", "jobs.jsonl", "Manifest.jsonl", "report.json",
              "run_settings.json", "recall.json", "timings.json", "keep.txt"):
        (tmp_path/f).write_text("x")
    reset_output_dir(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_preview", "keep.txt"]
    assert (tmp_path/"_preview"/"f.png").exists()


def test_reset_output_dir_missing_dir_is_noop(tmp_path):
    reset_output_dir(tmp_path/"absent")


def test_json_lines_progress_format():
    buf = io.StringIO(); p = JsonLinesProgress(buf)
    p.stage("masks", 2); p.advance(); p.event("done")
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert lines == [{"stage": "masks", "done": 0, "total": 2},
                     {"stage": "masks", "done": 1, "total": 2}, {"event": "done"}]


def test_progress_stage_resets_done_and_event_fields_korean():
    buf = io.StringIO(); p = JsonLinesProgress(buf)
    p.stage("a", 3); p.advance(2); p.stage("b", 5); p.advance(3)
    p.event("log", message="완료")
    assert "완료" in buf.getvalue()
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert lines[-2] == {"stage": "b", "done": 3, "total": 5}
    assert lines[-1] == {"event": "log", "message": "완료"}


def test_null_progress_is_silent():
    p = NullProgress(); p.stage("a", 1); p.advance(); p.event("x", y=1)
