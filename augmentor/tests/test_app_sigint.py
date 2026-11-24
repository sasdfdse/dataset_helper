import os
import signal
import time

import pytest

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no DISPLAY")


@needs_display
def test_ctrl_c_closes_running_mainloop():
    import tkinter as tk
    from gui.app import install_sigint
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    old = signal.getsignal(signal.SIGINT)
    destroyed = []
    root.bind("<Destroy>", lambda e: destroyed.append(e.widget is root) if e.widget is root else None)
    try:
        install_sigint(root)
        # send a real SIGINT while mainloop runs (same as Ctrl+C in a terminal)
        root.after(300, lambda: os.kill(os.getpid(), signal.SIGINT))
        safety = root.after(5000, root.destroy)       # if the handler never fires, this ends the test as a failure
        start = time.monotonic()
        root.mainloop()
        assert destroyed == [True]
        assert time.monotonic() - start < 3, "Ctrl+C로 바로 닫혀야 한다"
        del safety
        # during cleanup the default handler (KeyboardInterrupt) is back
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, old)
        try:
            root.destroy()
        except tk.TclError:
            pass
