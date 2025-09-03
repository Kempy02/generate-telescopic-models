# --------------------------------------------------
# Progress bar for long-running operations (estimation)
# --------------------------------------------------

import time, sys
from multiprocessing import Process, Event

_pb_evt = None
_pb_proc = None

def _fmt_time(s: float) -> str:
    s = max(0.0, s)
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

def _progress_worker(total_seconds: float, bar_len: int, tick: float, stop_evt: Event):
    start = time.perf_counter()
    while not stop_evt.is_set():
        now = time.perf_counter()
        elapsed = now - start
        frac = elapsed / total_seconds if total_seconds > 0 else 1.0
        if frac >= 1.0 and not stop_evt.is_set():
            frac = 0.99  # hold at 99% until main signals done

        filled = int(bar_len * frac)
        bar = "█" * filled + " " * (bar_len - filled)
        pct = int(frac * 100)
        remaining = max(0.0, total_seconds - elapsed)
        speed = (bar_len / total_seconds) if total_seconds > 0 else 0.0

        sys.stdout.write(f"\r{pct:3d}%|{bar}| {_fmt_time(elapsed)}<{_fmt_time(remaining)}, {speed:5.2f} it/s")
        sys.stdout.flush()
        time.sleep(tick)

    # finalize at 100%
    elapsed = time.perf_counter() - start
    sys.stdout.write(f"\r100%|{'█'*bar_len}| {_fmt_time(elapsed)}<00:00, {(bar_len/total_seconds) if total_seconds>0 else 0.0:5.2f} it/s\n")
    sys.stdout.flush()

def start_progress_bar(estimated_seconds: float, bar_len: int = 40, update_every: float = 0.1):
    """Start a conda-like progress bar in a separate process."""
    global _pb_evt, _pb_proc
    _pb_evt = Event()
    _pb_proc = Process(target=_progress_worker, args=(
        float(estimated_seconds), int(bar_len), float(update_every), _pb_evt
    ), daemon=True)
    _pb_proc.start()

def stop_progress_bar():
    """Stop the bar and print a final 100% line."""
    global _pb_evt, _pb_proc
    if _pb_evt is not None:
        _pb_evt.set()
    if _pb_proc is not None:
        _pb_proc.join()
    _pb_evt = None
    _pb_proc = None