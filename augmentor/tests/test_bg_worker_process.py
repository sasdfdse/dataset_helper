import sys
import threading

from gui.worker_process import WorkerProcess, parse_event


def test_parse_event():
    assert parse_event('{"stage": "masks", "done": 1, "total": 2}\n') == {"stage": "masks", "done": 1, "total": 2}
    assert parse_event("Loading checkpoint shards: 50%") is None
    assert parse_event("[1, 2]") is None
    assert parse_event("") is None


def test_worker_process_reports_events_and_exit(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text('import json,sys\nprint(json.dumps({"stage":"masks","done":0,"total":1}))\n'
                      'print("noise")\nprint(json.dumps({"event":"done"}))\nsys.stderr.write("last line\\n\\n")\nsys.exit(0)\n')
    events, exits, done = [], [], threading.Event()
    wp = WorkerProcess([], events.append, lambda c, e: (exits.append((c, e)), done.set()),
                       command=[sys.executable, str(script)])
    wp.start()
    assert done.wait(10)
    assert events == [{"stage": "masks", "done": 0, "total": 1}, {"event": "done"}]
    assert exits == [(0, "last line")]
    assert not wp.running


def test_worker_process_stop_terminates(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(60)\n")
    done = threading.Event()
    wp = WorkerProcess([], lambda e: None, lambda c, e: done.set(), command=[sys.executable, str(script)])
    wp.start()
    assert wp.running
    wp.stop()
    assert done.wait(10) and not wp.running


def test_stop_is_safe_when_not_running_or_already_exited(tmp_path):
    script = tmp_path / "quick.py"
    script.write_text("pass\n")
    done = threading.Event()
    wp = WorkerProcess([], lambda e: None, lambda c, e: done.set(), command=[sys.executable, str(script)])
    wp.stop()                     # before start
    assert not wp.running
    wp.start()
    assert done.wait(10)
    wp.stop()                     # already stopped
    assert not wp.running


def test_nonzero_exit_and_stderr_flood_does_not_deadlock(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text('import sys\nfor i in range(20000):\n    sys.stderr.write("x"*100+"\\n")\n'
                      'sys.stderr.write("boom\\n")\nsys.exit(5)\n')
    exits, done = [], threading.Event()
    wp = WorkerProcess([], lambda e: None, lambda c, e: (exits.append((c, e)), done.set()),
                       command=[sys.executable, str(script)])
    wp.start()
    assert done.wait(20)
    assert exits == [(5, "boom")]


def test_callback_exception_does_not_break_reader(tmp_path):
    script = tmp_path / "w.py"
    script.write_text('import json\nprint(json.dumps({"event":"warning","message":"a"}))\n'
                      'print(json.dumps({"event":"done"}))\n')
    seen, done = [], threading.Event()

    def on_event(ev):
        seen.append(ev)
        raise RuntimeError("callback bug")

    wp = WorkerProcess([], on_event, lambda c, e: done.set(), command=[sys.executable, str(script)])
    wp.start()
    assert done.wait(10) and len(seen) == 2
