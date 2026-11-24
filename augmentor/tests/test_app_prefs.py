"""기본 폴더 / 마지막 경로 기억 (앱을 꺼도 유지)."""
import os
from pathlib import Path

import pytest

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no DISPLAY")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "cfg" / "ui_state.json")
    from gui import app_prefs
    return app_prefs


def test_default_root_persists_and_drives_start_dir(prefs, tmp_path):
    root = tmp_path / "datasets"
    root.mkdir()
    assert prefs.default_root() is None and prefs.start_dir() == str(Path.home())
    prefs.set_default_root(root)
    assert prefs.prefs_path() == tmp_path / "cfg" / "app_prefs.json" and prefs.prefs_path().is_file()
    assert prefs.default_root() == root and prefs.start_dir() == str(root)
    # a filled field opens that folder (or the file's parent)
    sub = root / "a" / "images"
    sub.mkdir(parents=True)
    assert prefs.start_dir(str(sub)) == str(sub)
    assert prefs.start_dir(str(sub / "x.jpg")) == str(sub)
    # a vanished default folder is ignored
    root2 = tmp_path / "gone"
    prefs.set_default_root(root2)
    assert prefs.default_root() is None
    prefs.set_default_root(None)
    assert prefs.default_root() is None


def test_dialogs_start_in_default_root_and_remember_last_dir(prefs, tmp_path, monkeypatch):
    from tkinter import filedialog
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir(), other.mkdir()
    prefs.set_default_root(root)
    seen = []
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: (seen.append(k["initialdir"]), str(other))[1])
    assert prefs.ask_directory("t") == str(other) and seen == [str(root)]
    prefs.set_default_root(None)
    prefs.ask_directory("t")
    assert seen[-1] == str(other)                   # no default folder: last opened folder
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **k: ())
    assert prefs.ask_open_files("t") == []
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: "")
    assert prefs.ask_directory("t") == ""


def test_paths_and_broken_file(prefs):
    assert prefs.recall_path("augment.images") == ""
    prefs.remember_path("augment.images", "/data/images")
    assert prefs.recall_path("augment.images") == "/data/images"
    prefs.prefs_path().write_text("{oops", encoding="utf-8")
    assert prefs.recall_path("augment.images") == "" and prefs.default_root() is None
    prefs.remember_path("video.input", "/v.webm")          # saving still works over a corrupt file
    assert prefs.recall_path("video.input") == "/v.webm"


@needs_display
def test_augment_and_video_tabs_restore_last_paths(prefs, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import filedialog
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(), labels.mkdir()
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.augment_tab import AugmentTab
        from gui.video_tab import VideoTab
        answers = iter([str(images), str(labels)])
        monkeypatch.setattr(filedialog, "askdirectory", lambda **k: next(answers))
        tab = AugmentTab(root)
        tab._browse_images()
        tab._browse_labels()
        tab.destroy()
        v = VideoTab(root)
        v._input_var.set("/videos/a.webm")
        v.destroy()
        # "restarting the app" refills the last paths
        tab2 = AugmentTab(root)
        assert tab2._images_var.get() == str(images) and tab2._labels_var.get() == str(labels)
        assert tab2._images_dir == images and tab2._labels_dir == labels
        assert VideoTab(root)._input_var.get() == "/videos/a.webm"
    finally:
        root.destroy()


@needs_display
def test_app_default_root_button(prefs, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import filedialog
    from gui import app as app_mod
    monkeypatch.setattr(app_mod.BackgroundTab, "start_hardware_probe", lambda self: None)
    chosen = tmp_path / "yolo_dataset"
    chosen.mkdir()
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: str(chosen))
    try:
        a = app_mod.App()
    except tk.TclError:
        pytest.skip("cannot open display")
    a.withdraw()
    try:
        assert a._root_btn.cget("text") == "📁 기본 폴더 지정"
        a._root_btn.invoke()
        assert a._root_btn.cget("text") == "📁 기본 폴더: yolo_dataset"
        assert prefs.default_root() == chosen
    finally:
        a.destroy()
